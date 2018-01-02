#include "server/proxy.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "util/testing.h"

using cert_trans::FilterHeaders;
using cert_trans::UrlFetcher;
using std::make_pair;
using std::shared_ptr;
using std::string;

class ProxyTest : public ::testing::Test {};


TEST_F(ProxyTest, TestFilterHeadersLeavesUnrelatedHeadersAlone) {
  UrlFetcher::Headers headers;
  headers.insert(make_pair("one", "1"));
  headers.insert(make_pair("two", "2"));
  headers.insert(make_pair("three", "3"));
  headers.insert(make_pair("four", "4"));

  UrlFetcher::Headers response(headers);

  FilterHeaders(&response);
  EXPECT_EQ(headers, response);
}


string GetAll(const UrlFetcher::Headers& headers, const string& key) {
  const auto range(headers.equal_range(key));
  std::string ret;
  for (auto it(range.first); it != range.second; ++it) {
    if (!ret.empty()) {
      ret += ", ";
    }
    ret += it->second;
  }
  return ret;
}


TEST_F(ProxyTest, TestFilterHeadersRemovesReferencedHeaders) {
  UrlFetcher::Headers expected;
  expected.insert(make_pair("one", "1"));
  expected.insert(make_pair("four", "4"));

  UrlFetcher::Headers response(expected);
  response.insert(make_pair("two", "2"));
  response.insert(make_pair("three", "3"));
  response.insert(make_pair("Connection", "two, three, wibble"));


  FilterHeaders(&response);
  EXPECT_EQ(expected.size(), response.size());
  EXPECT_EQ(GetAll(expected, "one"), GetAll(response, "one"));
  EXPECT_EQ(GetAll(expected, "four"), GetAll(response, "four"));
}


TEST_F(ProxyTest, TestFilterHeadersHandlesMultipleReferencedHeaders) {
  UrlFetcher::Headers expected;
  expected.insert(make_pair("one", "1a"));
  expected.insert(make_pair("four", "4"));

  UrlFetcher::Headers response(expected);
  response.insert(make_pair("two", "2a"));
  response.insert(make_pair("two", "2b"));
  response.insert(make_pair("three", "3"));
  response.insert(make_pair("Connection", "two, three, wibble"));


  FilterHeaders(&response);
  EXPECT_EQ(expected.size(), response.size());
  EXPECT_EQ(GetAll(expected, "one"), GetAll(response, "one"));
  EXPECT_EQ(GetAll(expected, "four"), GetAll(response, "four"));
}


TEST_F(ProxyTest, TestFilterHeadersHandlesMultipleConnectionHeaders) {
  UrlFetcher::Headers expected;
  expected.insert(make_pair("one", "1a"));
  expected.insert(make_pair("four", "4"));

  UrlFetcher::Headers response(expected);
  response.insert(make_pair("two", "2"));
  response.insert(make_pair("three", "3"));
  response.insert(make_pair("Connection", "two"));
  response.insert(make_pair("Connection", "three"));
  response.insert(make_pair("Connection", "wibble"));


  FilterHeaders(&response);
  EXPECT_EQ(expected.size(), response.size());
  EXPECT_EQ(GetAll(expected, "one"), GetAll(response, "one"));
  EXPECT_EQ(GetAll(expected, "four"), GetAll(response, "four"));
}


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
