/* -*- mode: c++; indent-tabs-mode: nil -*- */

#ifndef UTIL_H
#define UTIL_H

#include <stdint.h>
#include <string>
#include <vector>

namespace util {

std::string HexString(const std::string& data);

std::string HexString(const std::string& data, char byte_delimiter);

std::string BinaryString(const std::string& hex_string);

bool ReadTextFile(const std::string& file, std::string* result);

bool ReadBinaryFile(const std::string& file, std::string* result);

// Write to a temporary file and return the filename, or an
// empty string on error.
std::string WriteTemporaryBinaryFile(const std::string& file_template,
                                     const std::string& data);

// Create a temporary directory, and return the dirname, or an
// empty string on error.
std::string CreateTemporaryDirectory(const std::string& dir_template);

uint64_t TimeInMilliseconds();

// Return a non-cryptographic random string. Caller needs to ensure
// srand() is called if needed.
std::string RandomString(size_t min_length, size_t max_length);

std::string FromBase64(const char* b64);

std::string ToBase64(const std::string& from);

std::vector<std::string> split(const std::string& in, char delim = ',');

}  // namespace util

#endif  // ndef UTIL_H
