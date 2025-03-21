import queue
import threading
import time
import webbrowser

# Open YouTube URL
webbrowser.open('https://www.youtube.com/watch?v=9yjZpBq1XBE&t=0s')

# Create a thread-safe queue
q = queue.Queue()

def producer():
    for i in range(5):
        item = f"Message {i}"
        print(f"Producing: {item}")
        q.put(item)  # Safe to use from multiple threads
        time.sleep(1)

def consumer():
    while True:
        item = q.get()  # Blocks until an item is available
        print(f"Consuming: {item}")
        q.task_done()
        if item == "Message 4":
            break

# Start producer and consumer threads
producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()