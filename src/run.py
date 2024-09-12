import sys

if __name__ == "__main__":
    context_window, delta, epsilon, iter_T = sys.argv[1:]
    print(f"context_window: {context_window}, delta: {delta}, epsilon: {epsilon}, iter_T: {iter_T}")