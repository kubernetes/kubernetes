#include "util/bignum.h"

#include <gtest/gtest.h>

#include "util/testing.h"


namespace cert_trans {


TEST(BigNumTest, TestDefaultCtor) {
  BigNum n;
  EXPECT_EQ(n, 0);
}


TEST(BigNumTest, TestWordCtor) {
  const int kValue(0x1234);
  BigNum n(kValue);
  EXPECT_EQ(n, kValue);
}


TEST(BigNumTest, TestCopyCtor) {
  const int kValue(0x1234);
  BigNum n(kValue);
  BigNum o(n);
  EXPECT_EQ(o, kValue);
}


TEST(BigNumTest, TestSubtractEqualsOperator) {
  const int kValue(0x1234);
  const int kDelta(10);
  BigNum n(kValue);
  n -= kDelta;
  EXPECT_EQ(n, kValue - kDelta);
}


TEST(BigNumTest, TestSubtractEqualsBigNumOperator) {
  const int kValue(0x1234);
  const int kDelta(10);
  BigNum n(kValue);
  BigNum d(kDelta);
  n -= d;
  EXPECT_EQ(n, kValue - kDelta);
}


TEST(BigNumTest, TestAddEqualsOperator) {
  const int kValue(0x1234);
  const int kDelta(10);
  BigNum n(kValue);
  n += kDelta;
  EXPECT_EQ(n, kValue + kDelta);
}


TEST(BigNumTest, TestAddEqualsBigNumOperator) {
  const int kValue(0x1234);
  const int kDelta(10);
  BigNum n(kValue);
  BigNum d(kDelta);
  n += d;
  EXPECT_EQ(n, kValue + kDelta);
}


TEST(BigNumTest, TestEqualsBigNumOperator) {
  const int kValue(0x1234);
  BigNum n(kValue);
  BigNum o(n);
  EXPECT_EQ(o, n);
}


TEST(BigNumTest, TestLShiftEqualsOperator) {
  const int kValue(0x1234);
  BigNum n(kValue);
  n <<= 2;
  EXPECT_EQ(n, kValue << 2);
}


TEST(BigNumTest, TestRShiftEqualsOperator) {
  const int kValue(0x1234);
  BigNum n(kValue);
  n >>= 2;
  EXPECT_EQ(n, kValue >> 2);
}


TEST(BigNumTest, TestBigNumEqualsBigNumOperator) {
  const int kValue(0x1234);
  BigNum n(kValue);
  BigNum o(kValue);
  EXPECT_EQ(n, o);
}


TEST(BigNumTest, TestBigNumEqualsIntOperator) {
  const int kValue(0x1234);
  BigNum n(kValue);
  EXPECT_EQ(n, kValue);
}


TEST(BigNumTest, TestBigNumLessThanBigNumOperator) {
  const int kValue(0x1234);
  BigNum n(kValue - 1);
  BigNum o(kValue);
  EXPECT_LT(n, o);
}


TEST(BigNumTest, TestBigNumLessThanIntOperator) {
  const int kValue(0x1234);
  BigNum n(kValue - 1);
  EXPECT_LT(n, kValue);
}


TEST(BigNumTest, TestBigNumGreaterThanBigNumOperator) {
  const int kValue(0x1234);
  BigNum n(kValue + 1);
  BigNum o(kValue);
  EXPECT_GT(n, o);
}


TEST(BigNumTest, TestBigNumGreaterThanIntOperator) {
  const int kValue(0x1234);
  BigNum n(kValue + 1);
  EXPECT_GT(n, kValue);
}


TEST(BigNumTest, TestBigNumAddBigNumOperator) {
  const int kValue(0x1234);
  BigNum n(kValue);
  BigNum o(kValue);
  EXPECT_EQ(kValue + kValue, n + o);
}


TEST(BigNumTest, TestBigNumAddIntOperator) {
  const int kValue(0x1234);
  BigNum n(kValue);
  EXPECT_EQ(kValue + kValue, n + kValue);
}


TEST(BigNumTest, TestBigNumSubtractBigNumOperator) {
  const int kValue(0x1234);
  const int kDelta(100);
  BigNum n(kValue);
  BigNum o(kDelta);
  EXPECT_EQ(kValue - kDelta, n - o);
}


TEST(BigNumTest, TestBigNumSubtractIntOperator) {
  const int kValue(0x1234);
  const int kDelta(100);
  BigNum n(kValue);
  EXPECT_EQ(kValue - kDelta, n - kDelta);
}


TEST(BigNumTest, TestBigNumLeftShiftOperator) {
  const int kValue(0x1234);
  BigNum n(kValue);
  EXPECT_EQ(kValue << 3, n << 3);
}


TEST(BigNumTest, TestBigNumRightShiftOperator) {
  const int kValue(0x1234);
  BigNum n(kValue);
  EXPECT_EQ(kValue >> 3, n >> 3);
}


}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
