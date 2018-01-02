#ifndef CERT_TRANS_UTIL_PROTOBUF_UTIL_H_
#define CERT_TRANS_UTIL_PROTOBUF_UTIL_H_

#include <ostream>

namespace google {
namespace protobuf {
namespace io {
class ZeroCopyOutputStream;
}  // namespace io

class MessageLite;
}  // namespace protobuf
}  // namespace google


namespace cert_trans {

// Implements the MessageLite.writeDelimitedTo() method from the Java proto
// API, which is strangly absent from the C++ library.
// This code was pinched from a response by Kenton Varda (ex Protobuf
// developer) to a question about this topic here:
//   http://stackoverflow.com/a/22927149
bool WriteDelimitedTo(const google::protobuf::MessageLite& message,
                      google::protobuf::io::ZeroCopyOutputStream* rawOutput);


bool WriteDelimitedToOstream(const google::protobuf::MessageLite& message,
                             std::ostream* os);


}  // namespace cert_trans


#endif  // CERT_TRANS_UTIL_PROTOBUF_UTIL_H_
