#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <set>
#include <string>
#include "log/etcd_consistent_store.h"

#include "base/notification.h"
#include "client/async_log_client.h"
#include "fetcher/remote_peer.h"
#include "log/log_verifier.h"
#include "log/test_db.h"
#include "log/test_signer.h"
#include "log/tree_signer.h"
#include "merkletree/merkle_verifier.h"
#include "merkletree/serial_hasher.h"
#include "monitoring/monitoring.h"
#include "net/mock_url_fetcher.h"
#include "proto/cert_serializer.h"
#include "util/fake_etcd.h"
#include "util/json_wrapper.h"
#include "util/mock_masterelection.h"
#include "util/sync_task.h"
#include "util/testing.h"
#include "util/thread_pool.h"

using ct::SignedTreeHead;
using std::bind;
using std::chrono::seconds;
using std::function;
using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::make_shared;
using std::map;
using std::ostream;
using std::set;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using testing::_;
using testing::InSequence;
using testing::Invoke;
using testing::NiceMock;
using testing::Return;
using util::Status;
using util::SyncTask;
using util::Task;

namespace ct {


ostream& operator<<(ostream& os, const SignedTreeHead& sth) {
  return os << sth.DebugString();
}


}  // namespace ct


namespace cert_trans {

DECLARE_int32(remote_peer_sth_refresh_interval_seconds);

const char kLogUrl[] = "https://example.com";
const char kInvalidSthReceivedMetric[] = "remote_peer_invalid_sths_received";


void HandleFetch(Status status, int status_code,
                 const UrlFetcher::Headers& headers, const string& body,
                 const UrlFetcher::Request& req, UrlFetcher::Response* resp,
                 Task* task) {
  resp->status_code = status_code;
  resp->headers = headers;
  resp->body = body;
  task->Return(status);
}


string Jsonify(const SignedTreeHead& sth) {
  JsonObject json_reply;
  json_reply.Add("tree_size", sth.tree_size());
  json_reply.Add("timestamp", sth.timestamp());
  json_reply.AddBase64("sha256_root_hash", sth.sha256_root_hash());
  json_reply.Add("tree_head_signature", sth.signature());
  LOG(INFO) << "Returning " << json_reply.ToString();
  return json_reply.ToString();
}


class RemotePeerTest : public ::testing::Test {
 public:
  RemotePeerTest()
      : base_(make_shared<libevent::Base>()),
        event_pump_(base_),
        etcd_client_(base_.get()),
        store_(base_.get(), &pool_, &etcd_client_, &election_, "/root", "id"),
        log_signer_(TestSigner::DefaultLogSigner()),
        tree_signer_(std::chrono::duration<double>(0), test_db_.db(),
                     unique_ptr<CompactMerkleTree>(
                         new CompactMerkleTree(new Sha256Hasher)),
                     &store_, log_signer_.get()),
        task_(&pool_) {
    FLAGS_remote_peer_sth_refresh_interval_seconds = 1;
    StoreInitialSthMetricValues();
  }

  ~RemotePeerTest() {
    task_.task()->Cancel();
    task_.Wait();
  }

  void CreatePeer() {
    peer_.reset(new RemotePeer(
        unique_ptr<AsyncLogClient>(
            new AsyncLogClient(&pool_, &fetcher_, kLogUrl)),
        unique_ptr<LogVerifier>(
            new LogVerifier(TestSigner::DefaultLogSigVerifier(),
                            new MerkleVerifier(new Sha256Hasher))),
        bind(&RemotePeerTest::OnNewSTH, this, _1), task_.task()));
  }

  void StoreInitialSthMetricValues() {
    set<const Metric*> metrics(Registry::Instance()->GetMetrics());
    for (const auto& m : metrics) {
      if (m->Name() == kInvalidSthReceivedMetric) {
        for (const auto& v : m->CurrentValues()) {
          ASSERT_EQ(static_cast<size_t>(1), v.first.size());
          invalid_sth_metric_values_[v.first[0]] = v.second.second;
        }
        return;
      }
    }
    LOG(FATAL) << "Couldn't find metric " << kInvalidSthReceivedMetric;
  }


  double GetInvalidSthMetricValue(const string& label) {
    set<const Metric*> metrics(Registry::Instance()->GetMetrics());
    for (const auto& m : metrics) {
      if (m->Name() == kInvalidSthReceivedMetric) {
        for (const auto& v : m->CurrentValues()) {
          if (v.first.size() == 1 && v.first[0] == label) {
            // Only interested in the delta from the start of the test.
            return v.second.second - invalid_sth_metric_values_[label];
          }
        }
        // no such label - return a default
        return 0;
      }
    }
    LOG(FATAL) << "Couldn't find metric " << kInvalidSthReceivedMetric;
  }

  MOCK_METHOD1(OnNewSTH, void(const SignedTreeHead& sth));

  void ExpectGetSth(const SignedTreeHead& return_sth, bool repeated = false) {
    if (!repeated) {
      EXPECT_CALL(fetcher_, Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                                    URL(string(kLogUrl) +
                                                        "/ct/v1/get-sth"),
                                                    _, ""),
                                  _, _))
          .WillOnce(
              Invoke(bind(&HandleFetch, Status::OK, 200, UrlFetcher::Headers{},
                          Jsonify(return_sth), _1, _2, _3)));
    } else {
      EXPECT_CALL(fetcher_, Fetch(IsUrlFetchRequest(UrlFetcher::Verb::GET,
                                                    URL(string(kLogUrl) +
                                                        "/ct/v1/get-sth"),
                                                    _, ""),
                                  _, _))
          .WillRepeatedly(
              Invoke(bind(&HandleFetch, Status::OK, 200, UrlFetcher::Headers{},
                          Jsonify(return_sth), _1, _2, _3)));
    }
  }

  void ReturnLatestSTH(const UrlFetcher::Request& req,
                       UrlFetcher::Response* resp, Task* task) {
    tree_signer_.UpdateTree();
    HandleFetch(Status::OK, 200, UrlFetcher::Headers{},
                Jsonify(tree_signer_.LatestSTH()), req, resp, task);
  }

 protected:
  shared_ptr<libevent::Base> base_;
  libevent::EventPumpThread event_pump_;
  FakeEtcdClient etcd_client_;
  TestDB<LevelDB> test_db_;
  ThreadPool pool_;
  NiceMock<MockMasterElection> election_;
  cert_trans::EtcdConsistentStore<LoggedEntry> store_;
  TestSigner test_signer_;
  unique_ptr<LogSigner> log_signer_;
  TreeSigner<LoggedEntry> tree_signer_;
  SyncTask task_;
  MockUrlFetcher fetcher_;
  unique_ptr<RemotePeer> peer_;
  map<string, double> invalid_sth_metric_values_;
};


MATCHER_P(EqualsSTH, sth, "") {
  return sth.timestamp() == arg.timestamp() &&
         sth.signature().DebugString() == arg.signature().DebugString() &&
         sth.version() == arg.version() &&
         sth.sha256_root_hash() == arg.sha256_root_hash() &&
         sth.tree_size() == arg.tree_size();
}


TEST_F(RemotePeerTest, RejectsSTHWithInvalidTimestamp) {
  tree_signer_.UpdateTree();

  const SignedTreeHead sth(tree_signer_.LatestSTH());

  SignedTreeHead modified_sth(sth);
  modified_sth.set_timestamp(modified_sth.timestamp() + 10000000);

  tree_signer_.UpdateTree();
  SignedTreeHead new_sth(tree_signer_.LatestSTH());

  {
    InSequence s;
    ExpectGetSth(sth);
    ExpectGetSth(modified_sth);
    ExpectGetSth(new_sth, true /* repeatedly */);
  }

  Notification notify;
  {
    InSequence t;
    EXPECT_CALL(*this, OnNewSTH(EqualsSTH(sth))).Times(1);
    EXPECT_CALL(*this, OnNewSTH(EqualsSTH(modified_sth))).Times(0);
    EXPECT_CALL(*this, OnNewSTH(EqualsSTH(new_sth)))
        .WillOnce(Invoke(bind(&Notification::Notify, &notify)));
  }

  CreatePeer();
  notify.WaitForNotificationWithTimeout(seconds(5));

  EXPECT_EQ(1, GetInvalidSthMetricValue("invalid_timestamp"));
  EXPECT_EQ(0, GetInvalidSthMetricValue("invalid_signature"));
}


TEST_F(RemotePeerTest, RejectsSTHWithInvalidSignature) {
  tree_signer_.UpdateTree();
  const SignedTreeHead sth(tree_signer_.LatestSTH());

  SignedTreeHead modified_sth(sth);
  modified_sth.mutable_signature()->set_signature("Autograph");

  tree_signer_.UpdateTree();
  SignedTreeHead new_sth(tree_signer_.LatestSTH());

  {
    InSequence s;
    ExpectGetSth(sth);
    ExpectGetSth(modified_sth);
    ExpectGetSth(new_sth, true /* repeatedly */);
  }

  Notification notify;
  {
    InSequence t;
    EXPECT_CALL(*this, OnNewSTH(EqualsSTH(sth))).Times(1);
    EXPECT_CALL(*this, OnNewSTH(EqualsSTH(modified_sth))).Times(0);
    EXPECT_CALL(*this, OnNewSTH(EqualsSTH(new_sth)))
        .WillOnce(Invoke(bind(&Notification::Notify, &notify)));
  }

  CreatePeer();
  notify.WaitForNotificationWithTimeout(seconds(5));

  EXPECT_EQ(0, GetInvalidSthMetricValue("invalid_timestamp"));
  EXPECT_EQ(1, GetInvalidSthMetricValue("invalid_signature"));
}


}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
