# Package: api_dispatcher

## Purpose
Manages the dispatching and execution of batched API calls from the scheduler. Handles concurrent execution with rate limiting and provides a queue-based mechanism for processing API calls.

## Key Types

### APIDispatcher (interface)
Defines the contract for dispatching API calls with methods like `Dispatch`, `Run`, and `RunCount`.

### apiDispatcher (struct)
The concrete implementation that manages:
- A priority queue of pending calls (organized by relevance/priority)
- Goroutine limiting to control concurrent API calls
- Signal channels for coordinating dispatch operations

### CallQueue
A priority queue for API calls sorted by relevance. Uses a heap-based implementation for efficient priority ordering.

### GoroutinesLimiter
Controls the number of concurrent goroutines executing API calls to prevent overwhelming the API server.

## Key Functions

- **New(client clientset.Interface, relevances)**: Creates a new dispatcher
- **Dispatch(call, onFinish)**: Adds an API call to the dispatch queue
- **Run(ctx)**: Main loop that processes queued API calls
- **Release(uid, relevance, onFinish)**: Signals completion of an API call

## Design Pattern
- Producer-consumer pattern with the APICacher as producer and APIDispatcher as consumer
- Uses relevance-based priority ordering (e.g., bindings before status patches)
- Implements backpressure via goroutine limiting
- Non-blocking dispatch with async completion notification
