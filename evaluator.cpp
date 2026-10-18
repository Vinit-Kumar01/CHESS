
#include <iostream>
#include <string>
using namespace std;

int main() {
    string board;

    cout << "Enter chessboard layout: ";
    cin >> board;

    int score = 0;

    for (char piece : board) {
        switch (piece) {
            case 'P': score += 100; break;
            case 'N': score += 300; break;
            case 'B': score += 300; break;
            case 'R': score += 500; break;
            case 'Q': score += 900; break;

            case 'p': score -= 100; break;
            case 'n': score -= 300; break;
            case 'b': score -= 300; break;
            case 'r': score -= 500; break;
            case 'q': score -= 900; break;
        }
    }

    cout << "Position score: " << score << endl;

    return 0;
}
