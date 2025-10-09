#ifndef _VALIDATE_H
#define _VALIDATE_H

#include <functional>
#include <regex>
#include <stdexcept>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#if !defined(_WIN32)
#include <arpa/inet.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>

// <windows.h> uses macros to #define a ton of symbols,
// many of which interfere with our code here and down
// the line in various extensions.
#undef DELETE
#undef ERROR
#undef GetMessage
#undef interface
#undef TRUE
#undef min

#endif

#include "google/protobuf/message.h"

namespace pgv {
using std::string;

class UnimplementedException : public std::runtime_error {
public:
  UnimplementedException() : std::runtime_error("not yet implemented") {}
  UnimplementedException(const std::string& message) : std::runtime_error(message) {}
  // Thrown by C++ validation code that is not yet implemented.
};

using ValidationMsg = std::string;

class BaseValidator {
public:
  /**
   * Validate/check a generic message object with a registered validator for the concrete message
   * type.
   * @param m supplies the message to check.
   * @param err supplies the place to return error information.
   * @return true if the validation passes OR there is no registered validator for the concrete
   *         message type. false is returned if validation explicitly fails.
   */
  static bool AbstractCheckMessage(const google::protobuf::Message& m, ValidationMsg* err) {
    // Polymorphic lookup is used to see if there is a matching concrete validator. If so, call it.
    // Otherwise return success.
    auto it = abstractValidators().find(std::type_index(typeid(m)));
    if (it == abstractValidators().end()) {
      return true;
    }
    return it->second(m, err);
  }

protected:
  // Used to implement AbstractCheckMessage() above. Every message that is linked into the binary
  // will register itself by type_index, allowing for polymorphic lookup later.
  static std::unordered_map<std::type_index,
                            std::function<bool(const google::protobuf::Message&, ValidationMsg*)>>&
  abstractValidators() {
    static auto* validator_map = new std::unordered_map<
        std::type_index, std::function<bool(const google::protobuf::Message&, ValidationMsg*)>>();
    return *validator_map;
  }
};

template <typename T> class Validator : public BaseValidator {
public:
  Validator(std::function<bool(const T&, ValidationMsg*)> check) : check_(check) {
    abstractValidators()[std::type_index(typeid(T))] = [this](const google::protobuf::Message& m,
                                                              ValidationMsg* err) -> bool {
      return check_(dynamic_cast<const T&>(m), err);
    };
  }

private:
  std::function<bool(const T&, ValidationMsg*)> check_;
};

static inline std::string String(const ValidationMsg& msg) { return std::string(msg); }

static inline bool IsPrefix(const string& maybe_prefix, const string& search_in) {
  return search_in.compare(0, maybe_prefix.size(), maybe_prefix) == 0;
}

static inline bool IsSuffix(const string& maybe_suffix, const string& search_in) {
  return maybe_suffix.size() <= search_in.size() &&
         search_in.compare(search_in.size() - maybe_suffix.size(), maybe_suffix.size(),
                           maybe_suffix) == 0;
}

static inline bool Contains(const string& search_in, const string& to_find) {
  return search_in.find(to_find) != string::npos;
}

static inline bool NotContains(const string& search_in, const string& to_find) {
  return !Contains(search_in, to_find);
}

static inline bool IsIpv4(const string& to_validate) {
  struct sockaddr_in sa;
  return !(inet_pton(AF_INET, to_validate.c_str(), &sa.sin_addr) < 1);
}

static inline bool IsIpv6(const string& to_validate) {
  struct sockaddr_in6 sa_six;
  return !(inet_pton(AF_INET6, to_validate.c_str(), &sa_six.sin6_addr) < 1);
}

static inline bool IsIp(const string& to_validate) {
  return IsIpv4(to_validate) || IsIpv6(to_validate);
}

static inline bool IsHostname(const string& to_validate) {
  if (to_validate.length() > 253) {
    return false;
  }

  const std::regex dot_regex{"\\."};
  const auto iter_end = std::sregex_token_iterator();
  auto iter = std::sregex_token_iterator(to_validate.begin(), to_validate.end(), dot_regex, -1);
  for (; iter != iter_end; ++iter) {
    const std::string& part = *iter;
    if (part.empty() || part.length() > 63) {
      return false;
    }
    if (part.at(0) == '-') {
      return false;
    }
    if (part.at(part.length() - 1) == '-') {
      return false;
    }
    for (const auto& character : part) {
      if ((character < 'A' || character > 'Z') && (character < 'a' || character > 'z') &&
          (character < '0' || character > '9') && character != '-') {
        return false;
      }
    }
  }

  return true;
}

namespace {

inline int OneCharLen(const char* src) {
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

inline int UTF8FirstLetterNumBytes(const char *utf8_str, int str_len) {
  if (str_len == 0)
    return 0;
  return OneCharLen(utf8_str);
}

inline size_t Utf8Len(const string& narrow_string) {
  const char* str_char = narrow_string.c_str();
  ptrdiff_t byte_len = narrow_string.length();
  size_t unicode_len = 0;
  int char_len = 1;
  while (byte_len > 0 && char_len > 0) {
    char_len = UTF8FirstLetterNumBytes(str_char, byte_len);
    str_char += char_len;
    byte_len -= char_len;
    ++unicode_len;
  }
  return unicode_len;
}

} // namespace

} // namespace pgv

#endif // _VALIDATE_H
