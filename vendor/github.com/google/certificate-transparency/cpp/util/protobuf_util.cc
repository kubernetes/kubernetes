#include "util/protobuf_util.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/message.h>

namespace cert_trans {

using google::protobuf::MessageLite;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::OstreamOutputStream;
using google::protobuf::io::ZeroCopyOutputStream;


// Implements the MessageLite.writeDelimitedTo() method from the Java proto
// API, which is strangely absent from the C++ library.
// This code was pinched from a response by Kenton Varda (ex Protobuf
// developer) to a question about this topic here:
//   http://stackoverflow.com/a/22927149
bool WriteDelimitedTo(const MessageLite& message,
                      ZeroCopyOutputStream* rawOutput) {
  // We create a new coded stream for each message.  Don't worry, this is fast.
  CodedOutputStream output(rawOutput);

  // Write the size.
  const int size = message.ByteSize();
  output.WriteVarint32(size);

  uint8_t* buffer = output.GetDirectBufferForNBytesAndAdvance(size);
  if (buffer != NULL) {
    // Optimization:  The message fits in one buffer, so use the faster
    // direct-to-array serialization path.
    message.SerializeWithCachedSizesToArray(buffer);
  } else {
    // Slightly-slower path when the message is multiple buffers.
    message.SerializeWithCachedSizes(&output);
    if (output.HadError())
      return false;
  }

  return true;
}


bool WriteDelimitedToOstream(const MessageLite& message, std::ostream* os) {
  OstreamOutputStream oos(os);
  return WriteDelimitedTo(message, &oos);
}


}  // namespace cert_trans
