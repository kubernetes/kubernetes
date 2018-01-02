#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <list>
#include <memory>
#include <string>

#include "net/mock_url_fetcher.h"
#include "util/etcd.h"
#include "util/json_wrapper.h"
#include "util/libevent_wrapper.h"
#include "util/status_test_util.h"
#include "util/sync_task.h"
#include "util/testing.h"

DECLARE_int32(etcd_watch_error_retry_delay_seconds);

namespace cert_trans {

using std::bind;
using std::chrono::seconds;
using std::list;
using std::make_pair;
using std::make_shared;
using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::unique_ptr;
using std::vector;
using testing::ElementsAre;
using testing::HasSubstr;
using testing::Invoke;
using testing::InSequence;
using testing::IsEmpty;
using testing::Pair;
using testing::StrCaseEq;
using testing::StrictMock;
using testing::_;
using util::Status;
using util::SyncTask;
using util::Task;
using util::testing::StatusIs;

namespace {

typedef UrlFetcher::Request FetchRequest;

const char kEntryKey[] = "/some/key";
const char kDirKey[] = "/some";
const char kGetJson[] =
    "{"
    "  \"action\": \"get\","
    "  \"node\": {"
    "    \"createdIndex\": 6,"
    "    \"key\": \"/some/key\","
    "    \"modifiedIndex\": 9,"
    "    \"value\": \"123\""
    "  }"
    "}";

const char kGetAllJson[] =
    "{"
    "  \"action\": \"get\","
    "  \"node\": {"
    "    \"createdIndex\": 1,"
    "    \"dir\": true,"
    "    \"key\": \"/some\","
    "    \"modifiedIndex\": 2,"
    "    \"nodes\": ["
    "      {"
    "        \"createdIndex\": 6,"
    "        \"key\": \"/some/key1\","
    "        \"modifiedIndex\": 9,"
    "        \"value\": \"123\""
    "      }, {"
    "        \"createdIndex\": 7,"
    "        \"key\": \"/some/key2\","
    "        \"modifiedIndex\": 7,"
    "        \"value\": \"456\""
    "      },"
    "    ]"
    "  }"
    "}";

const char kCreateJson[] =
    "{"
    "  \"action\": \"set\","
    "  \"node\": {"
    "    \"createdIndex\": 7,"
    "    \"key\": \"/some/key\","
    "    \"modifiedIndex\": 7,"
    "    \"value\": \"123\""
    "  }"
    "}";

const char kUpdateJson[] =
    "{"
    "  \"action\": \"set\","
    "  \"node\": {"
    "    \"createdIndex\": 5,"
    "    \"key\": \"/some/key\","
    "    \"modifiedIndex\": 6,"
    "    \"value\": \"123\""
    "  },"
    "  \"prevNode\": {"
    "    \"createdIndex\": 5,"
    "    \"key\": \"/some/key\","
    "    \"modifiedIndex\": 5,"
    "    \"value\": \"old\""
    "  }"
    "}";

const char kDeleteJson[] =
    "{"
    "  \"action\": \"delete\","
    "  \"node\": {"
    "    \"createdIndex\": 5,"
    "    \"key\": \"/some/key\","
    "    \"modifiedIndex\": 6,"
    "  },"
    "  \"prevNode\": {"
    "    \"createdIndex\": 5,"
    "    \"key\": \"/some/key\","
    "    \"modifiedIndex\": 5,"
    "    \"value\": \"123\""
    "  }"
    "}";

const char kKeyNotFoundJson[] =
    "{"
    "   \"index\" : 17,"
    "   \"message\" : \"Key not found\","
    "   \"errorCode\" : 100,"
    "   \"cause\" : \"/testdir/345\""
    "}";

const char kKeyAlreadyExistsJson[] =
    "{"
    "   \"index\" : 18,"
    "   \"errorCode\" : 105,"
    "   \"message\" : \"Key already exists\","
    "   \"cause\" : \"/a\""
    "}";

const char kCompareFailedJson[] =
    "{"
    "   \"errorCode\": 101,"
    "   \"message\": \"Compare failed\","
    "   \"cause\": \"[two != one]\","
    "   \"index\": 8"
    "}";

const char kStoreStatsJson[] =
    "{"
    "   \"setsFail\" : 1,"
    "   \"getsSuccess\" : 2,"
    "   \"watchers\" : 3,"
    "   \"expireCount\" : 4,"
    "   \"createFail\" : 5,"
    "   \"setsSuccess\" : 6,"
    "   \"compareAndDeleteFail\" : 7,"
    "   \"createSuccess\" : 8,"
    "   \"deleteFail\" : 9,"
    "   \"compareAndSwapSuccess\" : 10,"
    "   \"compareAndSwapFail\" : 11,"
    "   \"compareAndDeleteSuccess\" : 12,"
    "   \"updateFail\" : 13,"
    "   \"deleteSuccess\" : 14,"
    "   \"updateSuccess\" : 15,"
    "   \"getsFail\" : 16"
    "}";

const char kVersionString[] = "ETCD VERSION";

const char kEtcdHost[] = "etcd.example.net";
const int kEtcdPort = 4242;

const char kEtcdHost2[] = "etcd2.example.net";
const int kEtcdPort2 = 5252;

const char kEtcdHost3[] = "etcd3.example.net";
const int kEtcdPort3 = 6262;

const char kDefaultSpace[] = "/v2/keys";


string GetEtcdUrl(const string& key, const string& key_space = kDefaultSpace,
                  const string& host = kEtcdHost,
                  const uint16_t port = kEtcdPort) {
  CHECK(!key.empty() && key[0] == '/') << "key isn't slash-prefixed: " << key;
  return "http://" + string(host) + ":" + to_string(port) + key_space + key;
}


void HandleFetch(Status status, int status_code,
                 const UrlFetcher::Headers& headers, const string& body,
                 const UrlFetcher::Request& req, UrlFetcher::Response* resp,
                 Task* task) {
  resp->status_code = status_code;
  resp->headers = headers;
  resp->body = body;
  task->Return(status);
}


class EtcdTest : public ::testing::Test {
 public:
  EtcdTest()
      : base_(make_shared<libevent::Base>()),
        pump_(base_),
        client_(base_.get(), &url_fetcher_, kEtcdHost, kEtcdPort) {
    ExpectVersionCalls(url_fetcher_, kEtcdHost, kEtcdPort);
    ExpectVersionCalls(url_fetcher_, kEtcdHost2, kEtcdPort2);
    ExpectVersionCalls(url_fetcher_, kEtcdHost3, kEtcdPort3);
  }

 protected:
  void ExpectVersionCalls(MockUrlFetcher& fetcher, const string& host,
                          uint16_t port) {
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                URL(GetEtcdUrl("/version", "", host, port)),
                                IsEmpty(), ""),
              _, _))
        .WillRepeatedly(
            Invoke(bind(HandleFetch, Status::OK, 200, UrlFetcher::Headers{},
                        kVersionString, _1, _2, _3)));
  }

  shared_ptr<JsonObject> MakeJson(const string& json) {
    return make_shared<JsonObject>(json);
  }

  const shared_ptr<libevent::Base> base_;
  MockUrlFetcher url_fetcher_;
  libevent::EventPumpThread pump_;
  EtcdClient client_;
};


typedef EtcdTest EtcdDeathTest;


TEST_F(EtcdTest, Get) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                      URL(GetEtcdUrl(kEntryKey) +
                                          "?consistent=true&quorum=true"),
                                      IsEmpty(), ""),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "11")},
                      kGetJson, _1, _2, _3)));

  SyncTask task(base_.get());
  EtcdClient::GetResponse resp;
  client_.Get(string(kEntryKey), &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(11, resp.etcd_index);
  EXPECT_EQ(9, resp.node.modified_index_);
  EXPECT_EQ("123", resp.node.value_);
}


TEST_F(EtcdTest, GetRecursive) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(
                        UrlFetcher::Verb::GET,
                        URL(GetEtcdUrl(kEntryKey) +
                            "?consistent=true&quorum=true&recursive=true"),
                        IsEmpty(), ""),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "11")},
                      kGetJson, _1, _2, _3)));

  SyncTask task(base_.get());
  EtcdClient::Request req(kEntryKey);
  req.recursive = true;
  EtcdClient::GetResponse resp;
  client_.Get(req, &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(11, resp.etcd_index);
  EXPECT_EQ(9, resp.node.modified_index_);
  EXPECT_EQ("123", resp.node.value_);
}


TEST_F(EtcdTest, GetForInvalidKey) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                      URL(GetEtcdUrl(kEntryKey) +
                                          "?consistent=true&quorum=true"),
                                      IsEmpty(), ""),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 404,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "17")},
                      kKeyNotFoundJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::GetResponse resp;
  client_.Get(string(kEntryKey), &resp, task.task());
  task.Wait();
  EXPECT_THAT(task.status(), StatusIs(util::error::NOT_FOUND,
                                      AllOf(HasSubstr("Key not found"),
                                            HasSubstr(string(kEntryKey)))));
}


TEST_F(EtcdTest, GetAll) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                      URL(GetEtcdUrl(kDirKey) +
                                          "?consistent=true&quorum=true"),
                                      IsEmpty(), ""),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kGetAllJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::GetResponse resp;
  client_.Get(string(kDirKey), &resp, task.task());
  task.Wait();
  ASSERT_OK(task);
  EXPECT_TRUE(resp.node.is_dir_);
  ASSERT_EQ(static_cast<size_t>(2), resp.node.nodes_.size());
  EXPECT_EQ(9, resp.node.nodes_[0].modified_index_);
  EXPECT_EQ("123", resp.node.nodes_[0].value_);
  EXPECT_EQ(7, resp.node.nodes_[1].modified_index_);
  EXPECT_EQ("456", resp.node.nodes_[1].value_);
}


TEST_F(EtcdTest, GetWaitTooOld) {
  const int kOldIndex(42);
  const int kNewIndex(2015);
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                      URL(GetEtcdUrl(kEntryKey) +
                                          "?consistent=true&quorum=false&"
                                          "recursive=true&wait=true&"
                                          "waitIndex=" +
                                          to_string(kOldIndex)),
                                      IsEmpty(), ""),
                    _, _))
      .WillOnce(Invoke(bind(
          HandleFetch, Status::OK, 404,
          UrlFetcher::Headers{make_pair("x-etcd-index", to_string(kNewIndex))},
          kKeyNotFoundJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Request req(kEntryKey);
  req.recursive = true;
  req.wait_index = kOldIndex;
  EtcdClient::GetResponse resp;
  client_.Get(req, &resp, task.task());
  task.Wait();
  EXPECT_THAT(task.status(), StatusIs(util::error::NOT_FOUND,
                                      AllOf(HasSubstr("Key not found"),
                                            HasSubstr(string(kEntryKey)))));
  EXPECT_EQ(kNewIndex, resp.etcd_index);
}


TEST_F(EtcdTest, Create) {
  EXPECT_CALL(
      url_fetcher_,
      Fetch(IsUrlFetchRequest(
                UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                ElementsAre(Pair(StrCaseEq("content-type"),
                                 "application/x-www-form-urlencoded")),
                "consistent=true&prevExist=false&quorum=true&value=123"),
            _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kCreateJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.Create(kEntryKey, "123", &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(7, resp.etcd_index);
}


TEST_F(EtcdTest, CreateFails) {
  EXPECT_CALL(
      url_fetcher_,
      Fetch(IsUrlFetchRequest(
                UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                ElementsAre(Pair(StrCaseEq("content-type"),
                                 "application/x-www-form-urlencoded")),
                "consistent=true&prevExist=false&quorum=true&value=123"),
            _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 412,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kKeyAlreadyExistsJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.Create(kEntryKey, "123", &resp, task.task());
  task.Wait();
  EXPECT_THAT(task.status(), StatusIs(util::error::FAILED_PRECONDITION,
                                      HasSubstr("Key already exists")));
}


TEST_F(EtcdTest, CreateWithTTL) {
  EXPECT_CALL(
      url_fetcher_,
      Fetch(
          IsUrlFetchRequest(
              UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
              ElementsAre(Pair(StrCaseEq("content-type"),
                               "application/x-www-form-urlencoded")),
              "consistent=true&prevExist=false&quorum=true&ttl=100&value=123"),
          _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kCreateJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.CreateWithTTL(kEntryKey, "123", seconds(100), &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(7, resp.etcd_index);
}


TEST_F(EtcdTest, CreateWithTTLFails) {
  EXPECT_CALL(
      url_fetcher_,
      Fetch(
          IsUrlFetchRequest(
              UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
              ElementsAre(Pair(StrCaseEq("content-type"),
                               "application/x-www-form-urlencoded")),
              "consistent=true&prevExist=false&quorum=true&ttl=100&value=123"),
          _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 412,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kKeyAlreadyExistsJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.CreateWithTTL(kEntryKey, "123", seconds(100), &resp, task.task());
  task.Wait();
  EXPECT_THAT(task.status(), StatusIs(util::error::FAILED_PRECONDITION,
                                      HasSubstr("Key already exists")));
}


TEST_F(EtcdTest, Update) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(
                        UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                        ElementsAre(Pair(StrCaseEq("content-type"),
                                         "application/x-www-form-urlencoded")),
                        "consistent=true&prevIndex=5&quorum=true&value=123"),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kUpdateJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.Update(kEntryKey, "123", 5, &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(6, resp.etcd_index);
}


TEST_F(EtcdTest, UpdateFails) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(
                        UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                        ElementsAre(Pair(StrCaseEq("content-type"),
                                         "application/x-www-form-urlencoded")),
                        "consistent=true&prevIndex=5&quorum=true&value=123"),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 412,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kCompareFailedJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.Update(kEntryKey, "123", 5, &resp, task.task());
  task.Wait();
  EXPECT_THAT(task.status(), StatusIs(util::error::FAILED_PRECONDITION,
                                      HasSubstr("Compare failed")));
}


TEST_F(EtcdTest, UpdateWithTTL) {
  EXPECT_CALL(
      url_fetcher_,
      Fetch(IsUrlFetchRequest(
                UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                ElementsAre(Pair(StrCaseEq("content-type"),
                                 "application/x-www-form-urlencoded")),
                "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
            _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kUpdateJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.UpdateWithTTL(kEntryKey, "123", seconds(100), 5, &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(6, resp.etcd_index);
}


TEST_F(EtcdTest, UpdateWithTTLFails) {
  EXPECT_CALL(
      url_fetcher_,
      Fetch(IsUrlFetchRequest(
                UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                ElementsAre(Pair(StrCaseEq("content-type"),
                                 "application/x-www-form-urlencoded")),
                "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
            _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 412,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kCompareFailedJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.UpdateWithTTL(kEntryKey, "123", seconds(100), 5, &resp, task.task());
  task.Wait();
  EXPECT_THAT(task.status(), StatusIs(util::error::FAILED_PRECONDITION,
                                      HasSubstr("Compare failed")));
}


TEST_F(EtcdTest, ForceSetForPreexistingKey) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(
                        UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                        ElementsAre(Pair(StrCaseEq("content-type"),
                                         "application/x-www-form-urlencoded")),
                        "consistent=true&quorum=true&value=123"),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kUpdateJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.ForceSet(kEntryKey, "123", &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(6, resp.etcd_index);
}


TEST_F(EtcdTest, ForceSetForNewKey) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(
                        UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                        ElementsAre(Pair(StrCaseEq("content-type"),
                                         "application/x-www-form-urlencoded")),
                        "consistent=true&quorum=true&value=123"),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kCreateJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.ForceSet(kEntryKey, "123", &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(7, resp.etcd_index);
}


TEST_F(EtcdTest, ForceSetWithTTLForPreexistingKey) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(
                        UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                        ElementsAre(Pair(StrCaseEq("content-type"),
                                         "application/x-www-form-urlencoded")),
                        "consistent=true&quorum=true&ttl=100&value=123"),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kUpdateJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.ForceSetWithTTL(kEntryKey, "123", std::chrono::duration<int>(100),
                          &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(6, resp.etcd_index);
}


TEST_F(EtcdTest, ForceSetWithTTLForNewKey) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(
                        UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                        ElementsAre(Pair(StrCaseEq("content-type"),
                                         "application/x-www-form-urlencoded")),
                        "consistent=true&quorum=true&ttl=100&value=123"),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kCreateJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::Response resp;
  client_.ForceSetWithTTL(kEntryKey, "123", seconds(100), &resp, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(7, resp.etcd_index);
}


TEST_F(EtcdTest, Delete) {
  EXPECT_CALL(
      url_fetcher_,
      Fetch(IsUrlFetchRequest(UrlFetcher::Verb::DELETE,
                              URL(GetEtcdUrl(kEntryKey) +
                                  "?consistent=true&prevIndex=5&quorum=true"),
                              IsEmpty(), ""),
            _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kDeleteJson, _1, _2, _3)));
  SyncTask task(base_.get());
  client_.Delete(kEntryKey, 5, task.task());
  task.Wait();
  EXPECT_OK(task);
}


TEST_F(EtcdTest, DeleteFails) {
  EXPECT_CALL(
      url_fetcher_,
      Fetch(IsUrlFetchRequest(UrlFetcher::Verb::DELETE,
                              URL(GetEtcdUrl(kEntryKey) +
                                  "?consistent=true&prevIndex=5&quorum=true"),
                              IsEmpty(), ""),
            _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 412,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kCompareFailedJson, _1, _2, _3)));
  SyncTask task(base_.get());
  client_.Delete(kEntryKey, 5, task.task());
  task.Wait();
  EXPECT_THAT(task.status(), StatusIs(util::error::FAILED_PRECONDITION,
                                      HasSubstr("Compare failed")));
}


TEST_F(EtcdTest, ForceDelete) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(UrlFetcher::Verb::DELETE,
                                      URL(GetEtcdUrl(kEntryKey) +
                                          "?consistent=true&quorum=true"),
                                      IsEmpty(), ""),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kDeleteJson, _1, _2, _3)));
  SyncTask task(base_.get());
  client_.ForceDelete(kEntryKey, task.task());
  task.Wait();
  EXPECT_OK(task);
}


TEST_F(EtcdTest, WatchInitialGetFailureCausesRetry) {
  FLAGS_etcd_watch_error_retry_delay_seconds = 1;
  {
    InSequence s;
    EXPECT_CALL(url_fetcher_,
                Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                        URL(GetEtcdUrl(kEntryKey) +
                                            "?consistent=true&quorum=true"),
                                        IsEmpty(), ""),
                      _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status(util::error::UNAVAILABLE, ""), 0,
                        UrlFetcher::Headers{}, "", _1, _2, _3)));
    EXPECT_CALL(url_fetcher_,
                Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                        URL(GetEtcdUrl(kEntryKey) +
                                            "?consistent=true&quorum=true"),
                                        IsEmpty(), ""),
                      _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 200,
                        UrlFetcher::Headers{make_pair("x-etcd-index", "9")},
                        kGetJson, _1, _2, _3)));
  }

  SyncTask task(base_.get());
  client_.Watch(kEntryKey,
                [&task](const vector<EtcdClient::Node>& updates) {
                  EXPECT_EQ(static_cast<size_t>(1), updates.size());
                  EXPECT_EQ(9, updates[0].modified_index_);
                  EXPECT_EQ("123", updates[0].value_);
                  task.Cancel();
                },
                task.task());
  task.Wait();
}


TEST_F(EtcdTest, WatchInitialGetFailureRetriesOnNextEtcd) {
  FLAGS_etcd_watch_error_retry_delay_seconds = 1;
  EtcdClient multi_client(base_.get(), &url_fetcher_,
                          {EtcdClient::HostPortPair(kEtcdHost, kEtcdPort),
                           EtcdClient::HostPortPair(kEtcdHost2, kEtcdPort2)});
  {
    InSequence s;
    EXPECT_CALL(url_fetcher_,
                Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                        URL(GetEtcdUrl(kEntryKey) +
                                            "?consistent=true&quorum=true"),
                                        IsEmpty(), ""),
                      _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status(util::error::UNAVAILABLE, ""), 0,
                        UrlFetcher::Headers{}, "", _1, _2, _3)));
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                URL(GetEtcdUrl(kEntryKey, kDefaultSpace,
                                               kEtcdHost2, kEtcdPort2) +
                                    "?consistent=true&quorum=true"),
                                IsEmpty(), ""),
              _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 200,
                        UrlFetcher::Headers{make_pair("x-etcd-index", "9")},
                        kGetJson, _1, _2, _3)));
  }

  SyncTask task(base_.get());
  multi_client.Watch(kEntryKey,
                     [&task](const vector<EtcdClient::Node>& updates) {
                       EXPECT_EQ(static_cast<size_t>(1), updates.size());
                       EXPECT_EQ(9, updates[0].modified_index_);
                       EXPECT_EQ("123", updates[0].value_);
                       task.Cancel();
                     },
                     task.task());
  task.Wait();
}


TEST_F(EtcdTest, WatchHangingGetTimeoutCausesRetry) {
  FLAGS_etcd_watch_error_retry_delay_seconds = 1;

  {
    InSequence s;
    EXPECT_CALL(url_fetcher_,
                Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                        URL(GetEtcdUrl(kEntryKey) +
                                            "?consistent=true&quorum=true"),
                                        IsEmpty(), ""),
                      _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 200,
                        UrlFetcher::Headers{make_pair("x-etcd-index", "9")},
                        kGetJson, _1, _2, _3)));
    EXPECT_CALL(url_fetcher_,
                Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                        URL(GetEtcdUrl(kEntryKey) +
                                            "?consistent=true&quorum=false" +
                                            "&recursive=true&wait=true" +
                                            "&waitIndex=10"),
                                        IsEmpty(), ""),
                      _, _))
        .WillOnce(Invoke(bind(HandleFetch,
                              Status(util::error::DEADLINE_EXCEEDED, ""), 0,
                              UrlFetcher::Headers{}, "", _1, _2, _3)));
    EXPECT_CALL(url_fetcher_,
                Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                        URL(GetEtcdUrl(kEntryKey) +
                                            "?consistent=true&quorum=false" +
                                            "&recursive=true&wait=true" +
                                            "&waitIndex=10"),
                                        IsEmpty(), ""),
                      _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 200,
                        UrlFetcher::Headers{make_pair("x-etcd-index", "9")},
                        kGetJson, _1, _2, _3)));
  }

  SyncTask task(base_.get());
  int num_updates(0);
  client_.Watch(kEntryKey,
                [&task,
                 &num_updates](const vector<EtcdClient::Node>& updates) {
                  EXPECT_EQ(static_cast<size_t>(1), updates.size());
                  EXPECT_EQ(9, updates[0].modified_index_);
                  EXPECT_EQ("123", updates[0].value_);
                  if (num_updates == 1) {
                    task.Cancel();
                  }
                  ++num_updates;
                },
                task.task());
  task.Wait();
}


TEST_F(EtcdTest, UnavailableEtcdRetriesOnNewServer) {
  EtcdClient multi_client(base_.get(), &url_fetcher_,
                          {EtcdClient::HostPortPair(kEtcdHost, kEtcdPort),
                           EtcdClient::HostPortPair(kEtcdHost2, kEtcdPort2)});

  {
    InSequence s;

    // first server fails
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(
                  UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                  ElementsAre(Pair(StrCaseEq("content-type"),
                                   "application/x-www-form-urlencoded")),
                  "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
              _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status(util::error::UNAVAILABLE, ""), 0,
                        UrlFetcher::Headers{}, "", _1, _2, _3)));
    // second does too
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(
                  UrlFetcher::Verb::PUT,
                  URL(GetEtcdUrl(kEntryKey, kDefaultSpace, kEtcdHost2,
                                 kEtcdPort2)),
                  ElementsAre(Pair(StrCaseEq("content-type"),
                                   "application/x-www-form-urlencoded")),
                  "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
              _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status(util::error::UNAVAILABLE, ""), 0,
                        UrlFetcher::Headers{}, "", _1, _2, _3)));
    // try with first again, fail.
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(
                  UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                  ElementsAre(Pair(StrCaseEq("content-type"),
                                   "application/x-www-form-urlencoded")),
                  "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
              _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status(util::error::UNAVAILABLE, ""), 0,
                        UrlFetcher::Headers{}, "", _1, _2, _3)));
    // finally the second etcd is up:
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(
                  UrlFetcher::Verb::PUT,
                  URL(GetEtcdUrl(kEntryKey, kDefaultSpace, kEtcdHost2,
                                 kEtcdPort2)),
                  ElementsAre(Pair(StrCaseEq("content-type"),
                                   "application/x-www-form-urlencoded")),
                  "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
              _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 200, UrlFetcher::Headers{},
                        kUpdateJson, _1, _2, _3)));
  }

  SyncTask task(base_.get());
  EtcdClient::Response resp;
  multi_client.UpdateWithTTL(kEntryKey, "123", seconds(100), 5, &resp,
                             task.task());
  task.Wait();
  EXPECT_OK(task.status());
}


TEST_F(EtcdTest, FollowsMasterChangeRedirectToNewHost) {
  // Excludes kEtcdHost3:
  EtcdClient multi_client(base_.get(), &url_fetcher_,
                          {EtcdClient::HostPortPair(kEtcdHost, kEtcdPort),
                           EtcdClient::HostPortPair(kEtcdHost2, kEtcdPort2)});

  {
    InSequence s;

    // first server redirects to a new master which was previously unknown to
    // the log server:
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(
                  UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                  ElementsAre(Pair(StrCaseEq("content-type"),
                                   "application/x-www-form-urlencoded")),
                  "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
              _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 307,
                        UrlFetcher::Headers{make_pair(
                            "location", GetEtcdUrl(kEntryKey, kDefaultSpace,
                                                   kEtcdHost3, kEtcdPort3))},
                        "", _1, _2, _3)));
    // log should attempt to contact the new:
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(
                  UrlFetcher::Verb::PUT,
                  URL(GetEtcdUrl(kEntryKey, kDefaultSpace, kEtcdHost3,
                                 kEtcdPort3)),
                  ElementsAre(Pair(StrCaseEq("content-type"),
                                   "application/x-www-form-urlencoded")),
                  "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
              _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 200, UrlFetcher::Headers{},
                        kUpdateJson, _1, _2, _3)));
  }

  SyncTask task(base_.get());
  EtcdClient::Response resp;
  multi_client.UpdateWithTTL(kEntryKey, "123", seconds(100), 5, &resp,
                             task.task());
  task.Wait();
  EXPECT_OK(task.status());
}


TEST_F(EtcdTest, FollowsMasterChangeRedirectToKnownHost) {
  EtcdClient multi_client(base_.get(), &url_fetcher_,
                          {EtcdClient::HostPortPair(kEtcdHost, kEtcdPort),
                           EtcdClient::HostPortPair(kEtcdHost2, kEtcdPort2)});

  {
    InSequence s;

    // first server redirects to a new master which was previously known to the
    // log server:
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(
                  UrlFetcher::Verb::PUT, URL(GetEtcdUrl(kEntryKey)),
                  ElementsAre(Pair(StrCaseEq("content-type"),
                                   "application/x-www-form-urlencoded")),
                  "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
              _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 307,
                        UrlFetcher::Headers{make_pair(
                            "location", GetEtcdUrl(kEntryKey, kDefaultSpace,
                                                   kEtcdHost2, kEtcdPort2))},
                        "", _1, _2, _3)));
    // log should attempt to contact the new master:
    EXPECT_CALL(
        url_fetcher_,
        Fetch(IsUrlFetchRequest(
                  UrlFetcher::Verb::PUT,
                  URL(GetEtcdUrl(kEntryKey, kDefaultSpace, kEtcdHost2,
                                 kEtcdPort2)),
                  ElementsAre(Pair(StrCaseEq("content-type"),
                                   "application/x-www-form-urlencoded")),
                  "consistent=true&prevIndex=5&quorum=true&ttl=100&value=123"),
              _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 200, UrlFetcher::Headers{},
                        kUpdateJson, _1, _2, _3)));
  }

  SyncTask task(base_.get());
  EtcdClient::Response resp;
  multi_client.UpdateWithTTL(kEntryKey, "123", seconds(100), 5, &resp,
                             task.task());
  task.Wait();
  EXPECT_OK(task.status());
}


TEST_F(EtcdTest, GetStoreStats) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                      URL(GetEtcdUrl("/store", "/v2/stats") +
                                          "?consistent=true&quorum=true"),
                                      IsEmpty(), ""),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "1")},
                      kStoreStatsJson, _1, _2, _3)));
  SyncTask task(base_.get());
  EtcdClient::StatsResponse response;
  client_.GetStoreStats(&response, task.task());
  task.Wait();
  EXPECT_OK(task);
  EXPECT_EQ(1, response.stats["setsFail"]);
  EXPECT_EQ(2, response.stats["getsSuccess"]);
  EXPECT_EQ(3, response.stats["watchers"]);
  EXPECT_EQ(4, response.stats["expireCount"]);
  EXPECT_EQ(5, response.stats["createFail"]);
  EXPECT_EQ(6, response.stats["setsSuccess"]);
  EXPECT_EQ(7, response.stats["compareAndDeleteFail"]);
  EXPECT_EQ(8, response.stats["createSuccess"]);
  EXPECT_EQ(9, response.stats["deleteFail"]);
  EXPECT_EQ(10, response.stats["compareAndSwapSuccess"]);
  EXPECT_EQ(11, response.stats["compareAndSwapFail"]);
  EXPECT_EQ(12, response.stats["compareAndDeleteSuccess"]);
  EXPECT_EQ(13, response.stats["updateFail"]);
  EXPECT_EQ(14, response.stats["deleteSuccess"]);
  EXPECT_EQ(15, response.stats["updateSuccess"]);
  EXPECT_EQ(16, response.stats["getsFail"]);
}


TEST_F(EtcdTest, SplitHosts) {
  const string hosts(string(kEtcdHost) + ":" + to_string(kEtcdPort) + "," +
                     kEtcdHost2 + ":" + to_string(kEtcdPort2));
  const list<EtcdClient::HostPortPair> split_hosts(SplitHosts(hosts));
  EXPECT_EQ(static_cast<size_t>(2), split_hosts.size());
  EXPECT_EQ(kEtcdHost, split_hosts.front().first);
  EXPECT_EQ(kEtcdPort, split_hosts.front().second);
  EXPECT_EQ(kEtcdHost2, split_hosts.back().first);
  EXPECT_EQ(kEtcdPort2, split_hosts.back().second);
}


TEST_F(EtcdTest, SplitHostsIgnoresBlanks) {
  const string hosts(string(kEtcdHost) + ":" + to_string(kEtcdPort) + ",," +
                     kEtcdHost2 + ":" + to_string(kEtcdPort2));
  const list<EtcdClient::HostPortPair> split_hosts(SplitHosts(hosts));
  EXPECT_EQ(static_cast<size_t>(2), split_hosts.size());
  EXPECT_EQ(kEtcdHost, split_hosts.front().first);
  EXPECT_EQ(kEtcdPort, split_hosts.front().second);
  EXPECT_EQ(kEtcdHost2, split_hosts.back().first);
  EXPECT_EQ(kEtcdPort2, split_hosts.back().second);
}


TEST_F(EtcdTest, SplitHostsIgnoresTrailingComma) {
  const string hosts(string(kEtcdHost) + ":" + to_string(kEtcdPort) + "," +
                     kEtcdHost2 + ":" + to_string(kEtcdPort2) + ",");
  const list<EtcdClient::HostPortPair> split_hosts(SplitHosts(hosts));
  EXPECT_EQ(static_cast<size_t>(2), split_hosts.size());
  EXPECT_EQ(kEtcdHost, split_hosts.front().first);
  EXPECT_EQ(kEtcdPort, split_hosts.front().second);
  EXPECT_EQ(kEtcdHost2, split_hosts.back().first);
  EXPECT_EQ(kEtcdPort2, split_hosts.back().second);
}


TEST_F(EtcdTest, SplitHostsIgnoresPrecedingComma) {
  const string hosts("," + string(kEtcdHost) + ":" + to_string(kEtcdPort) +
                     "," + kEtcdHost2 + ":" + to_string(kEtcdPort2));
  const list<EtcdClient::HostPortPair> split_hosts(SplitHosts(hosts));
  EXPECT_EQ(static_cast<size_t>(2), split_hosts.size());
  EXPECT_EQ(kEtcdHost, split_hosts.front().first);
  EXPECT_EQ(kEtcdPort, split_hosts.front().second);
  EXPECT_EQ(kEtcdHost2, split_hosts.back().first);
  EXPECT_EQ(kEtcdPort2, split_hosts.back().second);
}


TEST_F(EtcdTest, SplitHostsWithAllBlanks) {
  const list<EtcdClient::HostPortPair> split_hosts(SplitHosts(",,,,"));
  EXPECT_EQ(static_cast<size_t>(0), split_hosts.size());
}


TEST_F(EtcdTest, SplitHostsWithEmptyString) {
  const list<EtcdClient::HostPortPair> split_hosts(SplitHosts(""));
  EXPECT_EQ(static_cast<size_t>(0), split_hosts.size());
}


TEST_F(EtcdDeathTest, SplitHostsWithInvalidHostPortString) {
  EXPECT_DEATH(SplitHosts("host:2:monkey"), "Invalid host:port string");
}


TEST_F(EtcdDeathTest, SplitHostsWithNegativePort) {
  EXPECT_DEATH(SplitHosts("host:-1"), "Port is <= 0");
}


TEST_F(EtcdDeathTest, SplitHostsWithOutOfRangePort) {
  EXPECT_DEATH(SplitHosts("host:65536"), "Port is > 65535");
}

TEST_F(EtcdDeathTest, SplitHostsWithQuotedUrlInList) {
  EXPECT_DEATH(SplitHosts("host:123,\"host:4002\", host:456"),
               "Invalid etcd_server url specified: \\\"host:4002\\\"");
}

TEST_F(EtcdDeathTest, SplitHostsWithQuotedUrl) {
  EXPECT_DEATH(SplitHosts("\"host:4001\""),
               "Invalid etcd_server url specified: \\\"host:4001\\\"");
}


// Test that if the SplitHosts validation failed the bad URL would still
// be caught
TEST_F(EtcdDeathTest, UrlRejectsQuotedValue) {
  EXPECT_DEATH(URL("\"host:4001\""), "URL invalid: \"host:4001\"");
}

TEST_F(EtcdTest, LogsVersion) {
  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                      URL(GetEtcdUrl(kEntryKey) +
                                          "?consistent=true&quorum=true"),
                                      IsEmpty(), ""),
                    _, _))
      .Times(2)
      .WillRepeatedly(
          Invoke(bind(HandleFetch, Status::OK, 200,
                      UrlFetcher::Headers{make_pair("x-etcd-index", "11")},
                      kGetJson, _1, _2, _3)));

  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                      URL(GetEtcdUrl("/version", "")),
                                      IsEmpty(), ""),
                    _, _))
      .WillOnce(
          Invoke(bind(HandleFetch, Status::OK, 200, UrlFetcher::Headers{},
                      kVersionString, _1, _2, _3)));

  // Version fetching is lazy, so we have to run some other request to kick
  // it off
  {
    SyncTask task(base_.get());
    EtcdClient::GetResponse resp;
    client_.Get(string(kEntryKey), &resp, task.task());
    task.Wait();
  }
  // However, a second call to the same etcd should not cause another version
  // request to be sent
  {
    SyncTask task(base_.get());
    EtcdClient::GetResponse resp;
    client_.Get(string(kEntryKey), &resp, task.task());
    task.Wait();
  }
}


TEST_F(EtcdTest, LogsVersionWhenChangingServer) {
  {
    InSequence s;
    EXPECT_CALL(url_fetcher_,
                Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                        URL(GetEtcdUrl(kEntryKey) +
                                            "?consistent=true&quorum=true"),
                                        IsEmpty(), ""),
                      _, _))
        .WillOnce(Invoke(bind(
            HandleFetch, Status::OK, 307,
            UrlFetcher::Headers{make_pair("x-etcd-index", "11"),
                                make_pair("location", GetEtcdUrl("/", ""))},
            kGetJson, _1, _2, _3)));

    EXPECT_CALL(url_fetcher_,
                Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                        URL(GetEtcdUrl(kEntryKey) +
                                            "?consistent=true&quorum=true"),
                                        IsEmpty(), ""),
                      _, _))
        .WillOnce(
            Invoke(bind(HandleFetch, Status::OK, 200,
                        UrlFetcher::Headers{make_pair("x-etcd-index", "11")},
                        kGetJson, _1, _2, _3)));
  }

  EXPECT_CALL(url_fetcher_,
              Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                      URL(GetEtcdUrl("/version", "")),
                                      IsEmpty(), ""),
                    _, _))
      .Times(2)
      .WillRepeatedly(
          Invoke(bind(HandleFetch, Status::OK, 200, UrlFetcher::Headers{},
                      kVersionString, _1, _2, _3)));

  // Version fetching is lazy, so we have to run some other request to kick
  // it off.
  // But the first response returns a redirect to another etcd master, so
  // when we follow that we should perform another version request.
  {
    SyncTask task(base_.get());
    EtcdClient::GetResponse resp;
    client_.Get(string(kEntryKey), &resp, task.task());
    task.Wait();
  }
}


}  // namespace
}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  return RUN_ALL_TESTS();
}
