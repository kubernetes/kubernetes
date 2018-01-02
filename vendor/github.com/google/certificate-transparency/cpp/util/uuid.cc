#include <iomanip>
#include <random>
#include <sstream>

#include "util/uuid.h"

using std::hex;
using std::mt19937;
using std::nouppercase;
using std::random_device;
using std::setw;
using std::setfill;
using std::string;
using std::stringstream;
using std::uniform_int_distribution;

namespace cert_trans {

string UUID4() {
  random_device rd;
  mt19937 twister(rd());
  uniform_int_distribution<uint32_t> distribution(0, UINT32_MAX);

  const uint32_t a((distribution(twister) & 0xFFFFFFFFUL));
  const uint32_t b((distribution(twister) & 0xFFFF0FFFUL) | 0x00004000UL);
  const uint32_t c((distribution(twister) & 0x3FFFFFFFUL) | 0x80000000UL);
  const uint32_t d((distribution(twister) & 0xFFFFFFFFUL));

  stringstream oss;
  oss << hex << nouppercase << setfill('0');

  oss << setw(8) << (a) << '-';
  oss << setw(4) << (b >> 16) << '-';
  oss << setw(4) << (b & 0xFFFF) << '-';
  oss << setw(4) << (c >> 16) << '-';
  oss << setw(4) << (c & 0xFFFF);
  oss << setw(8) << d;

  return oss.str();
}


}  // namespace cert_trans
