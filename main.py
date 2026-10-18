
import pygame
import chess
import math
import random
import threading
import time

pygame.init()

WIDTH = 640
BOARD_HEIGHT = 640
MESSAGE_HEIGHT = 100
WINDOW_HEIGHT = BOARD_HEIGHT + MESSAGE_HEIGHT
SQUARE = WIDTH // 8

screen = pygame.display.set_mode((WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Chess Game")

board = chess.Board()
game_mode = None
selected_square = None
legal_destinations = []
last_move = None
warning_message = ""
warning_timer = 0

# Computer difficulty settings: deeper search and a longer thinking budget
# make the opponent stronger, while the minimum delay makes moves feel natural.
DIFFICULTIES = {
    "Easy":   {"depth": 2, "think_time": 0.25, "min_delay": 0.80},
    "Medium": {"depth": 3, "think_time": 0.65, "min_delay": 1.00},
    "Hard":   {"depth": 4, "think_time": 1.25, "min_delay": 1.20},
    "Expert": {"depth": 5, "think_time": 2.00, "min_delay": 1.45},
}
difficulty_names = list(DIFFICULTIES.keys())
difficulty = "Medium"

# The engine searches on a copied board in a background thread so the
# Pygame window stays responsive. Results are applied only if still current.
ai_thread = None
ai_thinking = False
ai_started_at = 0.0
ai_generation = 0
ai_result = None
ai_result_lock = threading.Lock()

# Modern wood-and-slate theme
LIGHT = (238, 232, 213)
DARK = (105, 139, 125)
HIGHLIGHT = (238, 202, 92)
MOVE_COLOR = (45, 150, 105)
CAPTURE_COLOR = (215, 75, 75)
CHECK_COLOR = (202, 73, 73)
BACKGROUND = (18, 25, 34)
TEXT_COLOR = (242, 245, 249)
WARNING_COLOR = (255, 125, 105)
PANEL = (29, 39, 51)
PANEL_LIGHT = (39, 52, 67)
ACCENT = (226, 181, 91)
MUTED = (163, 176, 190)

PIECES = {
    "K": "♔", "Q": "♕", "R": "♖",
    "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜",
    "b": "♝", "n": "♞", "p": "♟",
}

VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

piece_font = pygame.font.SysFont("segoeuisymbol", 66)
message_font = pygame.font.SysFont("Arial", 22, bold=True)
small_font = pygame.font.SysFont("Arial", 16)
title_font = pygame.font.SysFont("Arial", 42, bold=True)
button_font = pygame.font.SysFont("Arial", 21, bold=True)
subtitle_font = pygame.font.SysFont("Arial", 17)
coordinate_font = pygame.font.SysFont("Arial", 12, bold=True)


MATE_SCORE = 100000
PIECE_VALUES = VALUES


class SearchTimeout(Exception):
    """Raised when a search reaches its time budget."""


class SearchContext:
    def __init__(self, deadline):
        self.deadline = deadline
        self.nodes = 0

    def visit(self):
        self.nodes += 1
        # Check frequently enough to respect the time budget without doing
        # an extra clock read on every single node.
        if (self.nodes & 63) == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout


def evaluate_position(position):
    """White-centric evaluation: material plus simple positional features."""
    score = 0
    piece_counts = {
        chess.WHITE: {piece_type: 0 for piece_type in VALUES},
        chess.BLACK: {piece_type: 0 for piece_type in VALUES},
    }

    # Count all pieces first so king evaluation does not depend on square order.
    for piece in position.piece_map().values():
        piece_counts[piece.color][piece.piece_type] += 1

    for square, piece in position.piece_map().items():
        color = piece.color
        sign = 1 if color == chess.WHITE else -1
        piece_type = piece.piece_type
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        relative_rank = rank_index if color == chess.WHITE else 7 - rank_index

        # Reward central control. Knights and bishops benefit most from it.
        center_distance = abs(3.5 - file_index) + abs(3.5 - rank_index)
        center_bonus = int((7 - center_distance) * 4)
        score += sign * VALUES[piece_type]

        if piece_type == chess.PAWN:
            score += sign * (relative_rank * 7)
            if file_index in (3, 4):
                score += sign * 8
        elif piece_type == chess.KNIGHT:
            score += sign * center_bonus * 3
        elif piece_type == chess.BISHOP:
            score += sign * center_bonus * 2
        elif piece_type == chess.QUEEN:
            score += sign * center_bonus
        elif piece_type == chess.ROOK:
            if relative_rank == 6:
                score += sign * 18
            # Reward rooks on files without any pawns.
            file_has_pawn = any(
                (pawn := position.piece_at(chess.square(file_index, r))) is not None
                and pawn.piece_type == chess.PAWN
                for r in range(8)
            )
            if not file_has_pawn:
                score += sign * 10
        elif piece_type == chess.KING:
            non_pawn_material = sum(
                piece_counts[color][kind] * VALUES[kind]
                for kind in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
            )
            if non_pawn_material < 1400:
                score += sign * center_bonus * 2
            else:
                home_rank = 0 if color == chess.WHITE else 7
                if rank_index == home_rank and file_index in (1, 2, 5, 6):
                    score += sign * 18

    # Bishop pair bonus and doubled-pawn penalty.
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        if piece_counts[color][chess.BISHOP] >= 2:
            score += sign * 28

        for file_index in range(8):
            pawns_on_file = sum(
                1
                for rank_index in range(8)
                if (piece := position.piece_at(chess.square(file_index, rank_index)))
                and piece.color == color
                and piece.piece_type == chess.PAWN
            )
            if pawns_on_file > 1:
                score -= sign * 12 * (pawns_on_file - 1)

    if position.is_check():
        score += -35 if position.turn == chess.WHITE else 35

    return score


def move_order_score(position, move):
    """Search promising captures and promotions before quiet moves."""
    score = 0
    moving_piece = position.piece_at(move.from_square)
    captured_piece = position.piece_at(move.to_square)

    if position.is_capture(move):
        if captured_piece is None:  # en passant
            captured_value = VALUES[chess.PAWN]
        else:
            captured_value = VALUES[captured_piece.piece_type]
        attacker_value = VALUES[moving_piece.piece_type] if moving_piece else 0
        score += 10000 + captured_value * 10 - attacker_value

    if move.promotion:
        score += 8000 + VALUES.get(move.promotion, 0)

    # Prefer castling slightly, as it usually improves king safety.
    if moving_piece and moving_piece.piece_type == chess.KING:
        if abs(chess.square_file(move.to_square) - chess.square_file(move.from_square)) == 2:
            score += 40

    return score


def ordered_moves(position, captures_only=False):
    moves = list(position.legal_moves)
    if captures_only:
        moves = [
            move for move in moves
            if position.is_capture(move) or move.promotion
        ]
    moves.sort(key=lambda move: move_order_score(position, move), reverse=True)
    return moves


def quiescence(position, alpha, beta, context, ply):
    """Extend leaf search through forcing captures to reduce horizon errors."""
    context.visit()

    if position.is_checkmate():
        return -MATE_SCORE + ply if position.turn == chess.WHITE else MATE_SCORE - ply
    if position.is_stalemate() or position.is_insufficient_material():
        return 0

    in_check = position.is_check()
    stand_pat = evaluate_position(position)

    if not in_check:
        if position.turn == chess.WHITE:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)

    # If in check, search all evasions; otherwise only captures/promotions.
    moves = ordered_moves(position, captures_only=not in_check)
    if not moves:
        return stand_pat

    if position.turn == chess.WHITE:
        best_score = -math.inf if in_check else stand_pat
        for move in moves:
            position.push(move)
            score = quiescence(position, alpha, beta, context, ply + 1)
            position.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score

    best_score = math.inf if in_check else stand_pat
    for move in moves:
        position.push(move)
        score = quiescence(position, alpha, beta, context, ply + 1)
        position.pop()
        best_score = min(best_score, score)
        beta = min(beta, best_score)
        if alpha >= beta:
            break
    return best_score


def minimax(position, depth, alpha, beta, context, ply=0):
    context.visit()

    if position.is_checkmate():
        return -MATE_SCORE + ply if position.turn == chess.WHITE else MATE_SCORE - ply
    if position.is_stalemate() or position.is_insufficient_material():
        return 0

    if depth <= 0:
        return quiescence(position, alpha, beta, context, ply)

    moves = ordered_moves(position)
    if not moves:
        return evaluate_position(position)

    if position.turn == chess.WHITE:
        best_score = -math.inf
        for move in moves:
            position.push(move)
            score = minimax(position, depth - 1, alpha, beta, context, ply + 1)
            position.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break
        return best_score

    best_score = math.inf
    for move in moves:
        position.push(move)
        score = minimax(position, depth - 1, alpha, beta, context, ply + 1)
        position.pop()
        best_score = min(best_score, score)
        beta = min(beta, best_score)
        if alpha >= beta:
            break
    return best_score


def search_root(position, depth, context):
    """Return the best move for the side to move at this search depth."""
    moves = ordered_moves(position)
    if not moves:
        return None, 0

    alpha = -math.inf
    beta = math.inf
    best_move = None
    best_score = -math.inf if position.turn == chess.WHITE else math.inf

    for move in moves:
        context.visit()
        position.push(move)
        score = minimax(position, depth - 1, alpha, beta, context, 1)
        position.pop()

        if position.turn == chess.WHITE:
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)

        if alpha >= beta:
            break

    return best_move, best_score


def choose_computer_move(position, level):
    """Iterative-deepening search with a time limit; safe to run in a thread."""
    settings = DIFFICULTIES[level]
    deadline = time.monotonic() + settings["think_time"]
    fallback_moves = list(position.legal_moves)
    if not fallback_moves:
        return None

    # A legal fallback is always available even if the first search times out.
    best_move = random.choice(fallback_moves) if level == "Easy" else fallback_moves[0]

    for depth in range(1, settings["depth"] + 1):
        context = SearchContext(deadline)
        try:
            candidate, score = search_root(position, depth, context)
            if candidate is not None:
                best_move = candidate

            # Easy mode occasionally chooses a different legal move to keep
            # the opponent beatable rather than playing perfectly every turn.
            if level == "Easy" and depth == settings["depth"]:
                alternatives = []
                for move in ordered_moves(position):
                    position.push(move)
                    move_score = minimax(
                        position, max(0, depth - 1),
                        -math.inf, math.inf, context, 1
                    )
                    position.pop()
                    if position.turn == chess.BLACK:
                        if move_score <= score + 90:
                            alternatives.append(move)
                    else:
                        if move_score >= score - 90:
                            alternatives.append(move)
                if alternatives:
                    best_move = random.choice(alternatives)
        except SearchTimeout:
            break

        if time.monotonic() >= deadline:
            break

    return best_move


def _computer_worker(position_copy, level, generation, starting_fen):
    global ai_result
    move = choose_computer_move(position_copy, level)
    with ai_result_lock:
        # Ignore a result if the player restarted, returned to the menu, or
        # started a newer search while this background thread was working.
        if generation == ai_generation:
            ai_result = (generation, starting_fen, move)


def start_computer_search():
    global ai_thread, ai_thinking, ai_started_at, ai_generation, ai_result

    if ai_thinking or board.is_game_over():
        return

    ai_generation += 1
    generation = ai_generation
    position_copy = board.copy(stack=True)
    starting_fen = position_copy.fen()
    level = difficulty
    ai_started_at = time.monotonic()
    ai_thinking = True

    with ai_result_lock:
        ai_result = None

    ai_thread = threading.Thread(
        target=_computer_worker,
        args=(position_copy, level, generation, starting_fen),
        daemon=True,
    )
    ai_thread.start()


def cancel_computer_search():
    global ai_thinking, ai_generation, ai_result
    ai_generation += 1
    ai_thinking = False
    with ai_result_lock:
        ai_result = None


def update_computer_search():
    """Apply a finished AI move only after its natural minimum delay."""
    global ai_thinking, ai_result, selected_square, legal_destinations, last_move

    if game_mode != "computer" or board.turn != chess.BLACK or board.is_game_over():
        return

    if not ai_thinking:
        start_computer_search()
        return

    with ai_result_lock:
        result = ai_result

    if result is None:
        return

    generation, starting_fen, move = result
    if generation != ai_generation or starting_fen != board.fen():
        return

    minimum_delay = DIFFICULTIES[difficulty]["min_delay"]
    if time.monotonic() - ai_started_at < minimum_delay:
        return

    if move is not None and move in board.legal_moves:
        board.push(move)
        last_move = move

    ai_thinking = False
    with ai_result_lock:
        ai_result = None

    selected_square = None
    legal_destinations = []
    show_warning("")

    if board.is_checkmate():
        show_warning("CHECKMATE! COMPUTER WINS!", None)
    elif board.is_check():
        show_warning("CHECK! Protect your king!", None)
    elif board.is_game_over():
        show_warning(get_game_message(), None)

def get_game_message():
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        return f"CHECKMATE! {winner} WINS!"

    if board.is_stalemate():
        return "DRAW! Stalemate."

    if board.is_insufficient_material():
        return "DRAW! Insufficient material."

    if board.is_seventyfive_moves():
        return "DRAW! 75-move rule."

    if board.is_fivefold_repetition():
        return "DRAW! Fivefold repetition."

    if board.is_check():
        player = "White" if board.turn == chess.WHITE else "Black"
        return f"CHECK! {player}, protect your king!"

    if board.is_game_over():
        return "GAME OVER! Draw."

    if game_mode == "computer":
        return "Your turn (White)" if board.turn == chess.WHITE else "Computer thinking..."

    return "White to move" if board.turn == chess.WHITE else "Black to move"


def show_warning(message, duration=3000):
    global warning_message, warning_timer

    warning_message = message

    if duration is None:
        warning_timer = 0
    elif message:
        warning_timer = pygame.time.get_ticks() + duration
    else:
        warning_timer = 0


def draw_vertical_gradient(surface, top_color, bottom_color):
    """Paint a smooth vertical gradient without external assets."""
    height = surface.get_height()
    width = surface.get_width()
    if height <= 1:
        surface.fill(top_color)
        return
    for y in range(height):
        ratio = y / (height - 1)
        color = tuple(
            int(top_color[i] * (1 - ratio) + bottom_color[i] * ratio)
            for i in range(3)
        )
        pygame.draw.line(surface, color, (0, y), (width, y))


def draw_button(rect, label, selected=False, hovered=False, compact=False):
    """Draw a shadowed, rounded button."""
    pygame.draw.rect(screen, (10, 14, 20), rect.move(0, 5), border_radius=12)
    if selected:
        fill, outline, text_color = (83, 137, 111), (139, 194, 151), (250, 255, 250)
        if hovered:
            fill = (98, 157, 127)
    elif hovered:
        fill, outline, text_color = (65, 83, 101), ACCENT, TEXT_COLOR
    else:
        fill, outline, text_color = PANEL_LIGHT, (61, 77, 94), TEXT_COLOR

    pygame.draw.rect(screen, fill, rect, border_radius=12)
    pygame.draw.rect(screen, outline, rect, width=2, border_radius=12)
    font = small_font if compact else button_font
    label_surface = font.render(label, True, text_color)
    screen.blit(label_surface, label_surface.get_rect(center=rect.center))


def draw_menu():
    draw_vertical_gradient(screen, (29, 43, 57), (13, 19, 28))
    pygame.draw.circle(screen, (34, 51, 65), (36, 80), 105, 2)
    pygame.draw.circle(screen, (34, 51, 65), (606, 640), 145, 2)
    pygame.draw.circle(screen, (30, 45, 58), (606, 640), 115, 2)

    title = title_font.render("CHESS", True, TEXT_COLOR)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, 105)))
    pygame.draw.rect(screen, ACCENT, pygame.Rect(WIDTH // 2 - 48, 137, 96, 4), border_radius=2)
    subtitle = subtitle_font.render("THINK AHEAD. PLAY SMART.", True, MUTED)
    screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, 164)))

    card = pygame.Rect(68, 205, 504, 385)
    pygame.draw.rect(screen, (9, 14, 21), card.move(0, 8), border_radius=22)
    pygame.draw.rect(screen, PANEL, card, border_radius=22)
    pygame.draw.rect(screen, (54, 70, 86), card, width=1, border_radius=22)

    mode_label = small_font.render("CHOOSE YOUR GAME", True, ACCENT)
    screen.blit(mode_label, mode_label.get_rect(center=(WIDTH // 2, 235)))

    buttons = [
        (pygame.Rect(125, 258, 390, 58), "PLAYER vs PLAYER"),
        (pygame.Rect(125, 330, 390, 58), "PLAYER vs COMPUTER"),
    ]
    mouse_pos = pygame.mouse.get_pos()
    for rect, label in buttons:
        draw_button(rect, label, hovered=rect.collidepoint(mouse_pos))

    level_label = small_font.render("COMPUTER DIFFICULTY", True, MUTED)
    screen.blit(level_label, level_label.get_rect(center=(WIDTH // 2, 418)))

    button_width, gap = 111, 8
    total_width = button_width * 4 + gap * 3
    start_x = (WIDTH - total_width) // 2
    for index, level in enumerate(difficulty_names):
        rect = pygame.Rect(start_x + index * (button_width + gap), 440, button_width, 46)
        draw_button(rect, level, selected=(level == difficulty),
                    hovered=rect.collidepoint(mouse_pos), compact=True)

    note = subtitle_font.render("Select a level, then start your match", True, MUTED)
    screen.blit(note, note.get_rect(center=(WIDTH // 2, 520)))
    hint = small_font.render("1–4: Difficulty     ESC: Quit", True, (125, 142, 158))
    screen.blit(hint, hint.get_rect(center=(WIDTH // 2, 558)))
    pygame.display.flip()

def draw_board():
    draw_vertical_gradient(screen, (24, 34, 45), (15, 21, 29))
    board_rect = pygame.Rect(0, 0, WIDTH, BOARD_HEIGHT)
    pygame.draw.rect(screen, (8, 12, 17), board_rect, width=5)
    pygame.draw.rect(screen, (52, 67, 80), board_rect, width=2)

    checked_king = board.king(board.turn) if board.is_check() else None

    for row in range(8):
        for col in range(8):
            square = chess.square(col, 7 - row)
            color = LIGHT if (row + col) % 2 == 0 else DARK
            rect = pygame.Rect(col * SQUARE, row * SQUARE, SQUARE, SQUARE)
            pygame.draw.rect(screen, color, rect)

            if last_move and square in (last_move.from_square, last_move.to_square):
                tint = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
                tint.fill((250, 198, 75, 85))
                screen.blit(tint, rect.topleft)

            if square == selected_square:
                pygame.draw.rect(screen, HIGHLIGHT, rect)
            elif square == checked_king:
                pygame.draw.rect(screen, CHECK_COLOR, rect)

            piece = board.piece_at(square)
            if piece:
                symbol = PIECES[piece.symbol()]
                center = rect.center
                if piece.color == chess.WHITE:
                    fill_color, outline_color = (255, 251, 237), (56, 69, 77)
                else:
                    fill_color, outline_color = (30, 39, 47), (232, 226, 205)

                shadow = piece_font.render(symbol, True, (0, 0, 0))
                screen.blit(shadow, shadow.get_rect(center=(center[0] + 2, center[1] + 3)))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    edge = piece_font.render(symbol, True, outline_color)
                    screen.blit(edge, edge.get_rect(center=(center[0] + dx, center[1] + dy)))
                piece_surface = piece_font.render(symbol, True, fill_color)
                screen.blit(piece_surface, piece_surface.get_rect(center=center))

            coord_color = (70, 87, 91) if (row + col) % 2 == 0 else (226, 234, 218)
            if col == 0:
                rank_surface = coordinate_font.render(str(8 - row), True, coord_color)
                screen.blit(rank_surface, (rect.x + 4, rect.y + 2))
            if row == 7:
                file_surface = coordinate_font.render(chr(ord("a") + col), True, coord_color)
                screen.blit(file_surface, (rect.right - file_surface.get_width() - 4,
                                           rect.bottom - file_surface.get_height() - 2))

    for destination in legal_destinations:
        col = chess.square_file(destination)
        row = 7 - chess.square_rank(destination)
        center = (col * SQUARE + SQUARE // 2, row * SQUARE + SQUARE // 2)
        if board.piece_at(destination):
            pygame.draw.circle(screen, CAPTURE_COLOR, center, 29, 4)
            pygame.draw.circle(screen, (255, 229, 220), center, 33, 1)
        else:
            pygame.draw.circle(screen, (25, 90, 69), center, 13)
            pygame.draw.circle(screen, (68, 191, 137), center, 10)

    footer_rect = pygame.Rect(0, BOARD_HEIGHT, WIDTH, MESSAGE_HEIGHT)
    pygame.draw.rect(screen, (19, 27, 37), footer_rect)
    pygame.draw.line(screen, (69, 85, 99), (0, BOARD_HEIGHT), (WIDTH, BOARD_HEIGHT), 2)

    display_message = warning_message or get_game_message()
    display_color = WARNING_COLOR if warning_message else TEXT_COLOR
    message_surface = message_font.render(display_message, True, display_color)
    screen.blit(message_surface, (16, BOARD_HEIGHT + 10))

    status = (f"Mode: Player vs Computer  •  Level: {difficulty}"
              if game_mode == "computer" else "Mode: Player vs Player")
    status_surface = small_font.render(status, True, ACCENT)
    screen.blit(status_surface, (16, BOARD_HEIGHT + 40))

    help_text = small_font.render(
        "Click to move   •   R: Restart   •   M: Menu   •   D: Difficulty   •   ESC: Quit",
        True, MUTED
    )
    screen.blit(help_text, (16, BOARD_HEIGHT + 67))
    pygame.display.flip()

def select_piece(square):
    global selected_square, legal_destinations

    piece = board.piece_at(square)

    if piece and piece.color == board.turn:
        selected_square = square
        legal_destinations = [
            move.to_square
            for move in board.legal_moves
            if move.from_square == square
        ]
    else:
        selected_square = None
        legal_destinations = []


def handle_click(position):
    global selected_square, legal_destinations, last_move

    x, y = position

    if not (0 <= x < WIDTH and 0 <= y < BOARD_HEIGHT):
        return

    if board.is_game_over():
        show_warning(get_game_message(), None)
        return

    if game_mode == "computer" and board.turn == chess.BLACK:
        return

    col = x // SQUARE
    row = y // SQUARE
    square = chess.square(col, 7 - row)

    if selected_square is None:
        piece = board.piece_at(square)

        if piece and piece.color == board.turn:
            select_piece(square)

            if board.is_check():
                player = "White" if board.turn == chess.WHITE else "Black"
                show_warning(f"CHECK! {player}, protect your king!", 3000)

        return

    piece = board.piece_at(selected_square)
    move = chess.Move(selected_square, square)

    if (
        piece
        and piece.piece_type == chess.PAWN
        and chess.square_rank(square) in (0, 7)
    ):
        move.promotion = chess.QUEEN

    if move in board.legal_moves:
        board.push(move)
        last_move = move
        selected_square = None
        legal_destinations = []
        show_warning("")

        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            show_warning(f"CHECKMATE! {winner} WINS!", None)

        elif board.is_check():
            player = "White" if board.turn == chess.WHITE else "Black"
            show_warning(f"CHECK! {player}, protect your king!", None)

        elif board.is_game_over():
            show_warning(get_game_message(), None)

    else:
        if board.is_check():
            show_warning("ILLEGAL MOVE! Protect your king!", 3000)

        new_piece = board.piece_at(square)

        if new_piece and new_piece.color == board.turn:
            select_piece(square)
        else:
            selected_square = None
            legal_destinations = []


clock = pygame.time.Clock()
running = True
in_menu = True

while running:
    if in_menu:
        draw_menu()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    difficulty = difficulty_names[event.key - pygame.K_1]

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                x, y = event.pos

                if pygame.Rect(140, 250, 360, 65).collidepoint(x, y):
                    cancel_computer_search()
                    game_mode = "pvp"
                    board.reset()
                    selected_square = None
                    legal_destinations = []
                    last_move = None
                    show_warning("")
                    in_menu = False

                elif pygame.Rect(140, 345, 360, 65).collidepoint(x, y):
                    cancel_computer_search()
                    game_mode = "computer"
                    board.reset()
                    selected_square = None
                    legal_destinations = []
                    last_move = None
                    show_warning("")
                    in_menu = False

                else:
                    button_width = 132
                    gap = 12
                    total_width = button_width * 4 + gap * 3
                    start_x = (WIDTH - total_width) // 2
                    for index, level in enumerate(difficulty_names):
                        rect = pygame.Rect(
                            start_x + index * (button_width + gap), 480,
                            button_width, 48
                        )
                        if rect.collidepoint(x, y):
                            difficulty = level
                            break

        clock.tick(60)
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

            elif event.key == pygame.K_m:
                cancel_computer_search()
                in_menu = True
                selected_square = None
                legal_destinations = []
                show_warning("")

            elif event.key == pygame.K_r:
                cancel_computer_search()
                board.reset()
                selected_square = None
                legal_destinations = []
                last_move = None
                show_warning("")

            elif event.key == pygame.K_d:
                current_index = difficulty_names.index(difficulty)
                difficulty = difficulty_names[(current_index + 1) % len(difficulty_names)]
                if ai_thinking:
                    cancel_computer_search()
                show_warning(f"Difficulty: {difficulty}", 1800)

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            handle_click(event.pos)

    update_computer_search()

    if warning_message and warning_timer:
        if pygame.time.get_ticks() >= warning_timer:
            show_warning("")

    draw_board()
    clock.tick(60)

cancel_computer_search()
pygame.quit()
