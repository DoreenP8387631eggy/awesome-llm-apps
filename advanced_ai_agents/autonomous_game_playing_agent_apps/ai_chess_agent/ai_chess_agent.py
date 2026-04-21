"""AI Chess Agent using LLM to play chess against a human opponent.

This module implements a chess-playing AI agent that uses a large language model
to analyze board positions and make strategic moves via the python-chess library.
"""

import os
import chess
import chess.svg
import streamlit as st
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a grandmaster-level chess engine. You will be given a chess board 
position in FEN notation along with the list of legal moves in UCI format. 
Your task is to select the single best move from the legal moves list.

Respond with ONLY the UCI move string (e.g., e2e4, g1f3, e1g1 for castling).
Do not include any explanation or additional text — just the raw UCI move."""


def get_ai_move(board: chess.Board) -> chess.Move:
    """Ask the LLM to pick the best legal move for the current board position."""
    legal_moves = [move.uci() for move in board.legal_moves]
    fen = board.fen()

    user_message = (
        f"Current board (FEN): {fen}\n"
        f"Legal moves (UCI): {', '.join(legal_moves)}\n"
        "Select the best move."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=10,
    )

    move_uci = response.choices[0].message.content.strip()

    # Validate the returned move is actually legal
    try:
        move = chess.Move.from_uci(move_uci)
        if move in board.legal_moves:
            return move
    except ValueError:
        pass

    # Fallback: return the first legal move if LLM response is invalid
    st.warning(f"LLM returned invalid move '{move_uci}'. Falling back to first legal move.")
    return list(board.legal_moves)[0]


def render_board(board: chess.Board) -> str:
    """Render the board as an SVG string."""
    last_move = board.peek() if board.move_stack else None
    return chess.svg.board(
        board,
        lastmove=last_move,
        size=400,
        colors={"square light": "#f0d9b5", "square dark": "#b58863"},
    )


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()
    if "game_over" not in st.session_state:
        st.session_state.game_over = False
    if "status_message" not in st.session_state:
        st.session_state.status_message = "Your turn! Enter a move in UCI format (e.g., e2e4)."
    if "player_color" not in st.session_state:
        st.session_state.player_color = chess.WHITE


def main():
    st.set_page_config(page_title="AI Chess Agent", page_icon="♟️", layout="centered")
    st.title("♟️ AI Chess Agent")
    st.caption("Powered by GPT-4o — Play chess against a grandmaster-level AI")

    init_session_state()

    board: chess.Board = st.session_state.board

    # Sidebar controls
    with st.sidebar:
        st.header("Game Controls")
        if st.button("🔄 New Game", use_container_width=True):
            st.session_state.board = chess.Board()
            st.session_state.game_over = False
            st.session_state.status_message = "Your turn! Enter a move in UCI format (e.g., e2e4)."
            st.rerun()

        st.markdown("---")
        st.markdown("**You play as:** White ⬜")
        st.markdown("**AI plays as:** Black ⬛")
        st.markdown("---")
        st.markdown("**Move format:** UCI (e.g., `e2e4`, `g1f3`, `e1g1`)")

    # Display board
    board_svg = render_board(board)
    st.image(board_svg.encode(), use_container_width=False, width=420)

    # Status message
    st.info(st.session_state.status_message)

    # Player move input
    if not st.session_state.game_over and board.turn == chess.WHITE:
        with st.form(key="move_form", clear_on_submit=True):
            player_move_uci = st.text_input(
                "Your move (UCI):",
                placeholder="e.g., e2e4",
                max_chars=5,
            )
            submitted = st.form_submit_button("Make Move", use_container_width=True)

        if submitted and player_move_uci:
            try:
                move = chess.Move.from_uci(player_move_uci.strip().lower())
                if move in board.legal_moves:
                    board.push(move)
                    st.session_state.status_message = f"You played: {player_move_uci}. AI is thinking..."

                    if board.is_game_over():
                        st.session_state.game_over = True
                        st.session_state.status_message = f"Game over! Result: {board.result()}"
                    else:
                        # AI responds
                        with st.spinner("AI is thinking..."):
                            ai_move = get_ai_move(board)
                        board.push(ai_move)
                        st.session_state.status_message = (
                            f"AI played: {ai_move.uci()}. Your turn!"
                        )
                        if board.is_game_over():
                            st.session_state.game_over = True
                            st.session_state.status_message = f"Game over! Result: {board.result()}"
                    st.rerun()
                else:
                    st.error("Illegal move. Please enter a valid UCI move.")
            except ValueError:
                st.error("Invalid move format. Use UCI notation (e.g., e2e4).")

    if st.session_state.game_over:
        st.success("The game has ended. Start a new game from the sidebar!")


if __name__ == "__main__":
    main()
