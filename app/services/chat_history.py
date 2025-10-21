from collections import defaultdict
from threading import Lock


# ---------------------------------------------------------------------------
# Thread-safe in-memory cache for chat contexts
# ---------------------------------------------------------------------------
_chat_context_cache = defaultdict(dict)
_cache_lock = Lock()


def set_chat_context(user_id: str, chat_id: str, context: str):
    """Save (or update) the current context for a given user/chat pair."""
    with _cache_lock:
        _chat_context_cache[user_id][chat_id] = context


def get_chat_context(user_id: str, chat_id: str) -> str:
    """Retrieve stored context for a given user/chat pair."""
    with _cache_lock:
        return _chat_context_cache[user_id].get(chat_id, "")


def clear_chat_context(user_id: str, chat_id: str):
    """Delete all context for a given chat under a specific user."""
    with _cache_lock:
        if chat_id in _chat_context_cache[user_id]:
            del _chat_context_cache[user_id][chat_id]


# ---------------------------------------------------------------------------
# Build a readable context summary from chat messages
# ---------------------------------------------------------------------------
def build_context_from_messages(messages: list, num_pairs: int = 3) -> str:
    """
    Build a formatted chat history context from the most recent conversation pairs.

    Args:
        messages (list): Chat history as list of dicts with 'role' and 'content' keys.
        num_pairs (int): Number of user/assistant exchanges to include.

    Returns:
        str: A structured summary of the selected conversation pairs.
    """
    if not messages:
        return ""

    pairs = []
    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            user_msg = messages[i]["content"].strip()
            assistant_msg = messages[i + 1]["content"].strip()
            pairs.append((user_msg, assistant_msg))
            i += 2
        else:
            i += 1

    if not pairs:
        return ""

    # Keep the last `num_pairs`
    selected_pairs = pairs[-num_pairs:]

    context_parts = []
    for idx, (user_msg, assistant_msg) in enumerate(selected_pairs, start=1):
        block = (
            f"Previous Conversation Pair {idx}\n"
            f"{'-'*30}\n"
            f"User: {user_msg}\n"
            f"Assistant: {assistant_msg}"
        )
        context_parts.append(block)

    full_context = (
        "Conversation History\n"
        + "=" * 25
        + "\n"
        + "\n\n".join(context_parts)
        + "\n"
        + "=" * 25
        + "\nEnd of Chat History\n"
    )

    return full_context


# ---------------------------------------------------------------------------
# Example usage for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "PDF content from OpenWebUI"},
        {"role": "user", "content": "Who is Einstein?"},
        {"role": "assistant", "content": "Albert Einstein was a theoretical physicist."},
        {"role": "user", "content": "Is he still alive?"},
        {"role": "assistant", "content": "No, he passed away in 1955."},
        {"role": "user", "content": "What is the theory of relativity?"},
    ]

    print(build_context_from_messages(messages, 5))