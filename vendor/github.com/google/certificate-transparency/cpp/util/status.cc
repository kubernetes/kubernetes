// Copyright 2013 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <sstream>

#include "util/status.h"

using ::std::ostream;
using ::std::string;

namespace util {

namespace {


const Status& GetOk() {
  static const Status status;
  return status;
}

const Status& GetCancelled() {
  static const Status status(::util::error::CANCELLED, "");
  return status;
}

const Status& GetUnknown() {
  static const Status status(::util::error::UNKNOWN, "");
  return status;
}


}  // namespace


Status::Status() : code_(::util::error::OK), message_("") {
}

Status::Status(::util::error::Code error, const string& error_message)
    : code_(error), message_(error_message) {
  if (code_ == ::util::error::OK) {
    message_.clear();
  }
}

Status::Status(const Status& other)
    : code_(other.code_), message_(other.message_) {
}

Status& Status::operator=(const Status& other) {
  code_ = other.code_;
  message_ = other.message_;
  return *this;
}

const Status& Status::OK = GetOk();
const Status& Status::CANCELLED = GetCancelled();
const Status& Status::UNKNOWN = GetUnknown();

string Status::ToString() const {
  if (code_ == ::util::error::OK) {
    return "OK";
  }

  std::ostringstream oss;
  oss << code_ << ": " << message_;
  return oss.str();
}

string ErrorCodeString(util::error::Code error) {
  switch (error) {
    case util::error::OK:
      return "OK";
    case util::error::CANCELLED:
      return "CANCELLED";
    case util::error::UNKNOWN:
      return "UNKNOWN";
    case util::error::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
    case util::error::DEADLINE_EXCEEDED:
      return "DEADLINE_EXCEEDED";
    case util::error::NOT_FOUND:
      return "NOT_FOUND";
    case util::error::ALREADY_EXISTS:
      return "ALREADY_EXISTS";
    case util::error::PERMISSION_DENIED:
      return "PERMISSION_DENIED";
    case util::error::RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
    case util::error::FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
    case util::error::ABORTED:
      return "ABORTED";
    case util::error::OUT_OF_RANGE:
      return "OUT_OF_RANGE";
    case util::error::UNIMPLEMENTED:
      return "UNIMPLEMENTED";
    case util::error::INTERNAL:
      return "INTERNAL";
    case util::error::UNAVAILABLE:
      return "UNAVAILABLE";
    case util::error::DATA_LOSS:
      return "DATA_LOSS";
  }
  // Avoid using a "default" in the switch, so that the compiler can
  // give us a warning, but still provide a fallback here.
  return std::to_string(error);
}

extern ostream& operator<<(ostream& os, util::error::Code code) {
  os << ErrorCodeString(code);
  return os;
}

extern ostream& operator<<(ostream& os, const Status& other) {
  os << other.ToString();
  return os;
}


}  // namespace util
