# Protocol Specification

The ttrpc protocol is client/server protocol to support multiple request streams
over a single connection with lightweight framing. The client represents the
process which initiated the underlying connection and the server is the process
which accepted the connection. The protocol is currently defined as
asymmetrical, with clients sending requests and servers sending responses. Both
clients and servers are able to send stream data. The roles are also used in
determining the stream identifiers, with client initiated streams using odd
number identifiers and server initiated using even number. The protocol may be
extended in the future to support server initiated streams, that is not
supported in the latest version.

## Purpose

The ttrpc protocol is designed to be lightweight and optimized for low latency
and reliable connections between processes on the same host. The protocol does
not include features for handling unreliable connections such as handshakes,
resets, pings, or flow control. The protocol is designed to make low-overhead
implementations as simple as possible. It is not intended as a suitable
replacement for HTTP2/3 over the network.

## Message Frame

Each Message Frame consists of a 10-byte message header followed
by message data. The data length and stream ID are both big-endian
4-byte unsigned integers. The message type is an unsigned 1-byte
integer. The flags are also an unsigned 1-byte integer and
use is defined by the message type.

    +---------------------------------------------------------------+
    |                       Data Length (32)                        |
    +---------------------------------------------------------------+
    |                        Stream ID (32)                         |
    +---------------+-----------------------------------------------+
    | Msg Type (8)  |
    +---------------+
    |   Flags (8)   |
    +---------------+-----------------------------------------------+
    |                           Data (*)                            |
    +---------------------------------------------------------------+

The Data Length field represents the number of bytes in the Data field. The
total frame size will always be Data Length + 10 bytes. The maximum data length
is 4MB and any larger size should be rejected. Due to the maximum data size
being less than 16MB, the first frame byte should always be zero. This first
byte should be considered reserved for future use.

The Stream ID must be odd for client initiated streams and even for server
initiated streams. Server initiated streams are not currently supported.

## Mesage Types

| Message Type | Name     | Description                      |
|--------------|----------|----------------------------------|
| 0x01         | Request  | Initiates stream                 |
| 0x02         | Response | Final stream data and terminates |
| 0x03         | Data     | Stream data                      |

### Request

The request message is used to initiate stream and send along request data for
properly routing and handling the stream. The stream may indicate unary without
any inbound or outbound stream data with only a response is expected on the
stream. The request may also indicate the stream is still open for more data and
no response is expected until data is finished. If the remote indicates the
stream is closed, the request may be considered non-unary but without anymore
stream data sent. In the case of `remote closed`, the remote still expects to
receive a response or stream data. For compatibility with non streaming clients,
a request with empty flags indicates a unary request.

#### Request Flags

| Flag | Name            | Description                                      |
|------|-----------------|--------------------------------------------------|
| 0x01 | `remote closed` | Non-unary, but no more data expected from remote |
| 0x02 | `remote open`   | Non-unary, remote is still sending data          |

### Response

The response message is used to end a stream with data, an empty response, or
an error. A response message is the only expected message after a unary request.
A non-unary request does not require a response message if the server is sending
back stream data. A non-unary stream may return a single response message but no
other stream data may follow.

#### Response Flags

No response flags are defined at this time, flags should be empty.

### Data

The data message is used to send data on an already initialized stream. Either
client or server may send data. A data message is not allowed on a unary stream.
A data message should not be sent after indicating `remote closed` to the peer.
The last data message on a stream must set the `remote closed` flag.

The `no data` flag is used to indicate that the data message does not include
any data. This is normally used with the `remote closed` flag to indicate the
stream is now closed without transmitting any data. Since ttrpc normally
transmits a single object per message, a zero length data message may be
interpreted as an empty object. For example, transmitting the number zero as a
protobuf message ends up with a data length of zero, but the message is still
considered data and should be processed.

#### Data Flags

| Flag | Name            | Description                       |
|------|-----------------|-----------------------------------|
| 0x01 | `remote closed` | No more data expected from remote |
| 0x04 | `no data`       | This message does not have data   |

## Streaming

All ttrpc requests use streams to transfer data. Unary streams will only have
two messages sent per stream, a request from a client and a response from the
server. Non-unary streams, however, may send any numbers of messages from the
client and the server. This makes stream management more complicated than unary
streams since both client and server need to track additional state. To keep
this management as simple as possible, ttrpc minimizes the number of states and
uses two flags instead of control frames. Each stream has two states while a
stream is still alive: `local closed` and `remote closed`. Each peer considers
local and remote from their own perspective and sets flags from the other peer's
perspective. For example, if a client sends a data frame with the
`remote closed` flag, that is indicating that the client is now `local closed`
and the server will be `remote closed`. A unary operation does not need to send
these flags since each received message always indicates `remote closed`. Once a
peer is both `local closed` and `remote closed`, the stream is considered
finished and may be cleaned up.

Due to the asymmetric nature of the current protocol, a client should
always be in the `local closed` state before `remote closed` and a server should
always be in the `remote closed` state before `local closed`. This happens
because the client is always initiating requests and a client always expects a
final response back from a server to indicate the initiated request has been
fulfilled. This may mean server sends a final empty response to finish a stream
even after it has already completed sending data before the client.

### Unary State Diagram

         +--------+                                    +--------+
         | Client |                                    | Server |
         +---+----+                                    +----+---+
             |               +---------+                    |
      local  >---------------+ Request +--------------------> remote
      closed |               +---------+                    | closed
             |                                              |
             |              +----------+                    |
    finished <--------------+ Response +--------------------< finished
             |              +----------+                    |
             |                                              |

### Non-Unary State Diagrams

RC: `remote closed` flag
RO: `remote open` flag

         +--------+                                    +--------+
         | Client |                                    | Server |
         +---+----+                                    +----+---+
             |             +--------------+                 |
             >-------------+ Request [RO] +----------------->
             |             +--------------+                 |
             |                                              |
             |                 +------+                     |
             >-----------------+ Data +--------------------->
             |                 +------+                     |
             |                                              |
             |               +-----------+                  |
      local  >---------------+ Data [RC] +------------------> remote
      closed |               +-----------+                  | closed
             |                                              |
             |              +----------+                    |
    finished <--------------+ Response +--------------------< finished
             |              +----------+                    |
             |                                              |

         +--------+                                    +--------+
         | Client |                                    | Server |
         +---+----+                                    +----+---+
             |             +--------------+                 |
      local  >-------------+ Request [RC] +-----------------> remote
      closed |             +--------------+                 | closed
             |                                              |
             |                 +------+                     |
             <-----------------+ Data +---------------------<
             |                 +------+                     |
             |                                              |
             |               +-----------+                  |
    finished <---------------+ Data [RC] +------------------< finished
             |               +-----------+                  |
             |                                              |

         +--------+                                    +--------+
         | Client |                                    | Server |
         +---+----+                                    +----+---+
             |             +--------------+                 |
             >-------------+ Request [RO] +----------------->
             |             +--------------+                 |
             |                                              |
             |                 +------+                     |
             >-----------------+ Data +--------------------->
             |                 +------+                     |
             |                                              |
             |                 +------+                     |
             <-----------------+ Data +---------------------<
             |                 +------+                     |
             |                                              |
             |                 +------+                     |
             >-----------------+ Data +--------------------->
             |                 +------+                     |
             |                                              |
             |               +-----------+                  |
      local  >---------------+ Data [RC] +------------------> remote
      closed |               +-----------+                  | closed
             |                                              |
             |                 +------+                     |
             <-----------------+ Data +---------------------<
             |                 +------+                     |
             |                                              |
             |               +-----------+                  |
    finished <---------------+ Data [RC] +------------------< finished
             |               +-----------+                  |
             |                                              |

## RPC

While this protocol is defined primarily to support Remote Procedure Calls, the
protocol does not define the request and response types beyond the messages
defined in the protocol. The implementation provides a default protobuf
definition of request and response which may be used for cross language rpc.
All implementations should at least define a request type which support
routing by procedure name and a response type which supports call status.

## Version History

| Version | Features            |
|---------|---------------------|
| 1.0     | Unary requests only |
| 1.2     | Streaming support   |
