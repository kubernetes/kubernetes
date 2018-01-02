#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include "fetcher/mock_continuous_fetcher.h"
#include "log/cluster_state_controller-inl.h"
#include "log/logged_entry.h"
#include "log/test_db.h"
#include "net/mock_url_fetcher.h"
#include "proto/cert_serializer.h"
#include "proto/ct.pb.h"
#include "util/fake_etcd.h"
#include "util/libevent_wrapper.h"
#include "util/mock_masterelection.h"
#include "util/testing.h"
#include "util/thread_pool.h"
#include "util/util.h"

using ct::ClusterConfig;
using ct::ClusterNodeState;
using ct::SignedTreeHead;
using std::make_shared;
using std::shared_ptr;
using std::string;
using std::vector;
using testing::AnyNumber;
using testing::NiceMock;
using testing::Return;
using testing::_;
using util::StatusOr;

namespace cert_trans {

const char kNodeId1[] = "node1";
const char kNodeId2[] = "node2";
const char kNodeId3[] = "node3";


class ClusterStateControllerTest : public ::testing::Test {
 public:
  // TODO: Some of the tests in this class rely on sleep() calls.
  // Ideally they should be waiting for defined conditions to avoid timing
  // races.

  // TODO(pphaneuf): The size of the thread pool is a bit of a magic
  // number... We have some callbacks that block, so it has to be "at
  // least this" (so we're setting it explicitly, in case your machine
  // doesn't have enough cores). We should hunt down those blocking
  // callbacks.
  ClusterStateControllerTest()
      : pool_(3),
        base_(make_shared<libevent::Base>()),
        pump_(base_),
        etcd_(base_.get()),
        store1_(new EtcdConsistentStore<LoggedEntry>(base_.get(), &pool_,
                                                     &etcd_, &election1_, "",
                                                     kNodeId1)),
        store2_(new EtcdConsistentStore<LoggedEntry>(base_.get(), &pool_,
                                                     &etcd_, &election2_, "",
                                                     kNodeId2)),
        store3_(new EtcdConsistentStore<LoggedEntry>(base_.get(), &pool_,
                                                     &etcd_, &election3_, "",
                                                     kNodeId3)),
        controller_(&pool_, base_, &url_fetcher_, test_db_.db(), store1_.get(),
                    &election1_, &fetcher_) {
    // There will be many calls to ContinuousFetcher::AddPeer during
    // this test, but this isn't what we're testing here, so just
    // ignore them.
    EXPECT_CALL(fetcher_, AddPeer(_, _)).Times(AnyNumber());

    // Set default cluster config:
    ct::ClusterConfig default_config;
    default_config.set_minimum_serving_nodes(1);
    default_config.set_minimum_serving_fraction(1);
    store1_->SetClusterConfig(default_config);

    controller_.SetNodeHostPort(kNodeId1, 9001);

    // Set up some handy STHs
    sth100_.set_tree_size(100);
    sth100_.set_timestamp(100);
    sth200_.set_tree_size(200);
    sth200_.set_timestamp(200);
    sth300_.set_tree_size(300);
    sth300_.set_timestamp(300);

    cns100_.set_hostname(kNodeId1);
    cns100_.set_log_port(9001);
    cns100_.mutable_newest_sth()->CopyFrom(sth100_);

    cns200_.set_hostname(kNodeId2);
    cns200_.set_log_port(9001);
    cns200_.mutable_newest_sth()->CopyFrom(sth200_);

    cns300_.set_hostname(kNodeId3);
    cns300_.set_log_port(9001);
    cns300_.mutable_newest_sth()->CopyFrom(sth300_);
  }

 protected:
  ct::ClusterNodeState GetLocalState() {
    return controller_.local_node_state_;
  }

  // TODO: This should probably return a util::StatusOr<ClusterNodeState>
  // rather than failing a CHECK if absent.
  ct::ClusterNodeState GetNodeStateView(const string& node_id) {
    auto it(controller_.all_peers_.find("/nodes/" + node_id));
    CHECK(it != controller_.all_peers_.end());
    return it->second->state();
  }

  static void SetClusterConfig(ConsistentStore<LoggedEntry>* store,
                               const int min_nodes,
                               const double min_fraction) {
    ClusterConfig config;
    config.set_minimum_serving_nodes(min_nodes);
    config.set_minimum_serving_fraction(min_fraction);
    CHECK(store->SetClusterConfig(config).ok());
  }


  SignedTreeHead sth100_, sth200_, sth300_;
  ClusterNodeState cns100_, cns200_, cns300_;

  ThreadPool pool_;
  shared_ptr<libevent::Base> base_;
  MockUrlFetcher url_fetcher_;
  MockContinuousFetcher fetcher_;
  libevent::EventPumpThread pump_;
  FakeEtcdClient etcd_;
  TestDB<FileDB> test_db_;
  NiceMock<MockMasterElection> election1_;
  NiceMock<MockMasterElection> election2_;
  NiceMock<MockMasterElection> election3_;
  std::unique_ptr<EtcdConsistentStore<LoggedEntry>> store1_;
  std::unique_ptr<EtcdConsistentStore<LoggedEntry>> store2_;
  std::unique_ptr<EtcdConsistentStore<LoggedEntry>> store3_;
  ClusterStateController<LoggedEntry> controller_;
};


typedef class EtcdConsistentStoreTest EtcdConsistentStoreDeathTest;


TEST_F(ClusterStateControllerTest, TestNewTreeHead) {
  ct::SignedTreeHead sth;
  sth.set_tree_size(234);
  controller_.NewTreeHead(sth);
  EXPECT_EQ(sth.DebugString(), GetLocalState().newest_sth().DebugString());
}


TEST_F(ClusterStateControllerTest, TestCalculateServingSTHAt50Percent) {
  NiceMock<MockMasterElection> election_is_master;
  EXPECT_CALL(election_is_master, IsMaster()).WillRepeatedly(Return(true));
  ClusterStateController<LoggedEntry> controller50(
      &pool_, base_, &url_fetcher_, test_db_.db(), store1_.get(),
      &election_is_master, &fetcher_);
  SetClusterConfig(store1_.get(), 1 /* nodes */, 0.5 /* fraction */);

  store1_->SetClusterNodeState(cns100_);
  sleep(1);
  util::StatusOr<SignedTreeHead> sth(controller50.GetCalculatedServingSTH());
  // Can serve sth1 because all nodes have it !
  EXPECT_EQ(sth100_.tree_size(), sth.ValueOrDie().tree_size());

  store2_->SetClusterNodeState(cns200_);
  sleep(1);
  // Can serve sth2 because 50% of nodes have it
  sth = controller50.GetCalculatedServingSTH();
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());

  store3_->SetClusterNodeState(cns300_);
  sleep(1);
  // Can serve sth2 because 66% of nodes have it (or higher)
  // Can't serve sth3 because only 33% of nodes cover it.
  sth = controller50.GetCalculatedServingSTH();
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());
}


TEST_F(ClusterStateControllerTest, TestCalculateServingSTHAt70Percent) {
  NiceMock<MockMasterElection> election_is_master;
  EXPECT_CALL(election_is_master, IsMaster()).WillRepeatedly(Return(true));
  ClusterStateController<LoggedEntry> controller70(
      &pool_, base_, &url_fetcher_, test_db_.db(), store1_.get(),
      &election_is_master, &fetcher_);
  SetClusterConfig(store1_.get(), 1 /* nodes */, 0.7 /* fraction */);
  store1_->SetClusterNodeState(cns100_);
  sleep(1);
  util::StatusOr<SignedTreeHead> sth(controller70.GetCalculatedServingSTH());
  // Can serve sth1 because all nodes have it !
  EXPECT_EQ(sth100_.tree_size(), sth.ValueOrDie().tree_size());

  store2_->SetClusterNodeState(cns200_);
  sleep(1);
  // Can still only serve sth1 because only 50% of nodes have sth2
  sth = controller70.GetCalculatedServingSTH();
  EXPECT_EQ(sth100_.tree_size(), sth.ValueOrDie().tree_size());

  store3_->SetClusterNodeState(cns300_);
  sleep(1);
  // Can still only serve sth1 because only 66% of nodes have sth2
  sth = controller70.GetCalculatedServingSTH();
  EXPECT_EQ(sth100_.tree_size(), sth.ValueOrDie().tree_size());
}


TEST_F(ClusterStateControllerTest,
       TestCalculateServingSTHAt60PercentTwoNodeMin) {
  NiceMock<MockMasterElection> election_is_master;
  EXPECT_CALL(election_is_master, IsMaster()).WillRepeatedly(Return(true));
  ClusterStateController<LoggedEntry> controller60(
      &pool_, base_, &url_fetcher_, test_db_.db(), store1_.get(),
      &election_is_master, &fetcher_);
  SetClusterConfig(store1_.get(), 2 /* nodes */, 0.6 /* fraction */);
  store1_->SetClusterNodeState(cns100_);
  sleep(1);
  util::StatusOr<SignedTreeHead> sth(controller60.GetCalculatedServingSTH());
  // Can't serve at all because not enough nodes
  EXPECT_FALSE(sth.ok());

  store2_->SetClusterNodeState(cns200_);
  sleep(1);
  // Can serve sth1 because there are two nodes, but < 60% coverage for sth2
  sth = controller60.GetCalculatedServingSTH();
  EXPECT_EQ(sth100_.tree_size(), sth.ValueOrDie().tree_size());

  store3_->SetClusterNodeState(cns300_);
  sleep(1);
  sth = controller60.GetCalculatedServingSTH();
  // Can serve sth2 because there are two out of three nodes with sth2 or above
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());
}


TEST_F(ClusterStateControllerTest, TestCalculateServingSTHAsClusterMoves) {
  NiceMock<MockMasterElection> election_is_master;
  EXPECT_CALL(election_is_master, IsMaster()).WillRepeatedly(Return(true));
  ClusterStateController<LoggedEntry> controller50(
      &pool_, base_, &url_fetcher_, test_db_.db(), store1_.get(),
      &election_is_master, &fetcher_);
  SetClusterConfig(store1_.get(), 1 /* nodes */, 0.5 /* fraction */);
  ct::ClusterNodeState node_state(cns100_);
  store1_->SetClusterNodeState(node_state);
  node_state.set_hostname(kNodeId2);
  store2_->SetClusterNodeState(node_state);
  node_state.set_hostname(kNodeId3);
  store3_->SetClusterNodeState(node_state);
  sleep(1);
  util::StatusOr<SignedTreeHead> sth(controller50.GetCalculatedServingSTH());
  EXPECT_EQ(sth100_.tree_size(), sth.ValueOrDie().tree_size());

  node_state = cns200_;
  node_state.set_hostname(kNodeId1);
  store1_->SetClusterNodeState(node_state);
  sleep(1);
  // Node1@200
  // Node2 and Node3 @100:
  // Still have to serve at sth100
  sth = controller50.GetCalculatedServingSTH();
  EXPECT_EQ(sth100_.tree_size(), sth.ValueOrDie().tree_size());

  node_state.set_hostname(kNodeId3);
  store3_->SetClusterNodeState(node_state);
  sleep(1);
  // Node1 and Node3 @200
  // Node2 @100:
  // Can serve at sth200
  sth = controller50.GetCalculatedServingSTH();
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());

  node_state = cns300_;
  node_state.set_hostname(kNodeId2);
  store2_->SetClusterNodeState(node_state);
  sleep(1);
  // Node1 and Node3 @200
  // Node2 @300:
  // Still have to serve at sth200
  sth = controller50.GetCalculatedServingSTH();
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());
}


TEST_F(ClusterStateControllerTest, TestKeepsNewerSTH) {
  store1_->SetClusterNodeState(cns100_);

  // Create a node with an identically sized but newer STH:
  SignedTreeHead newer_sth(sth100_);
  newer_sth.set_timestamp(newer_sth.timestamp() + 1);
  ClusterNodeState newer_cns;
  newer_cns.set_hostname("somenode.example.net");
  newer_cns.set_log_port(9001);
  *newer_cns.mutable_newest_sth() = newer_sth;
  store2_->SetClusterNodeState(newer_cns);
  sleep(1);

  util::StatusOr<SignedTreeHead> sth(controller_.GetCalculatedServingSTH());
  EXPECT_EQ(newer_sth.tree_size(), sth.ValueOrDie().tree_size());
  EXPECT_EQ(newer_sth.timestamp(), sth.ValueOrDie().timestamp());
}


TEST_F(ClusterStateControllerTest, TestCannotSelectSmallerSTH) {
  NiceMock<MockMasterElection> election_is_master;
  EXPECT_CALL(election_is_master, IsMaster()).WillRepeatedly(Return(true));
  ClusterStateController<LoggedEntry> controller50(
      &pool_, base_, &url_fetcher_, test_db_.db(), store1_.get(),
      &election_is_master, &fetcher_);
  SetClusterConfig(store1_.get(), 1 /* nodes */, 0.5 /* fraction */);

  ct::ClusterNodeState node_state(cns200_);
  node_state.set_hostname(kNodeId1);
  store1_->SetClusterNodeState(node_state);
  node_state.set_hostname(kNodeId2);
  store2_->SetClusterNodeState(node_state);
  node_state.set_hostname(kNodeId3);
  store3_->SetClusterNodeState(node_state);
  sleep(1);
  util::StatusOr<SignedTreeHead> sth(controller50.GetCalculatedServingSTH());
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());

  node_state = cns100_;
  node_state.set_hostname(kNodeId1);
  store1_->SetClusterNodeState(node_state);
  sleep(1);
  // Node1@100
  // Node2 and Node3 @200:
  // Still have to serve at sth200
  sth = controller50.GetCalculatedServingSTH();
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());

  node_state.set_hostname(kNodeId3);
  store3_->SetClusterNodeState(node_state);
  sleep(1);
  // Node1 and Node3 @100
  // Node2 @200
  // But cannot select an earlier STH than the one we last served with, so must
  // stick with sth200:
  sth = controller50.GetCalculatedServingSTH();
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());

  node_state.set_hostname(kNodeId2);
  store2_->SetClusterNodeState(node_state);
  sleep(1);
  // Still have to serve at sth200
  sth = controller50.GetCalculatedServingSTH();
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());
}


TEST_F(ClusterStateControllerTest, TestUsesLargestSTHWithIdenticalTimestamp) {
  NiceMock<MockMasterElection> election_is_master;
  EXPECT_CALL(election_is_master, IsMaster()).WillRepeatedly(Return(true));
  ClusterStateController<LoggedEntry> controller50(
      &pool_, base_, &url_fetcher_, test_db_.db(), store1_.get(),
      &election_is_master, &fetcher_);
  SetClusterConfig(store1_.get(), 1 /* nodes */, 0.5 /* fraction */);

  ClusterNodeState cns1;
  cns1.set_hostname(kNodeId1);
  cns1.set_log_port(9001);
  cns1.mutable_newest_sth()->set_timestamp(1000);
  cns1.mutable_newest_sth()->set_tree_size(1000);
  store1_->SetClusterNodeState(cns1);

  ClusterNodeState cns2(cns1);
  cns2.set_hostname(kNodeId2);
  cns2.set_log_port(9001);
  cns2.mutable_newest_sth()->set_timestamp(1000);
  cns2.mutable_newest_sth()->set_tree_size(1001);
  store2_->SetClusterNodeState(cns2);

  ClusterNodeState cns3;
  cns3.set_hostname(kNodeId3);
  cns3.set_log_port(9001);
  cns3.mutable_newest_sth()->set_timestamp(1004);
  cns3.mutable_newest_sth()->set_tree_size(999);
  store3_->SetClusterNodeState(cns3);
  sleep(2);

  util::StatusOr<SignedTreeHead> sth(controller50.GetCalculatedServingSTH());
  EXPECT_EQ(cns2.newest_sth().tree_size(), sth.ValueOrDie().tree_size());
  EXPECT_EQ(cns2.newest_sth().timestamp(), sth.ValueOrDie().timestamp());
}


TEST_F(ClusterStateControllerTest, TestDoesNotReuseSTHTimestamp) {
  NiceMock<MockMasterElection> election_is_master;
  EXPECT_CALL(election_is_master, IsMaster()).WillRepeatedly(Return(true));
  ClusterStateController<LoggedEntry> controller50(
      &pool_, base_, &url_fetcher_, test_db_.db(), store1_.get(),
      &election_is_master, &fetcher_);
  SetClusterConfig(store1_.get(), 3 /* nodes */, 1 /* fraction */);

  ClusterNodeState cns1;
  cns1.set_hostname(kNodeId1);
  cns1.set_log_port(9001);
  cns1.mutable_newest_sth()->set_timestamp(1002);
  cns1.mutable_newest_sth()->set_tree_size(10);
  store1_->SetClusterNodeState(cns1);

  ClusterNodeState cns2(cns1);
  cns2.set_hostname(kNodeId2);
  cns2.set_log_port(9001);
  cns2.mutable_newest_sth()->set_timestamp(1000);
  cns2.mutable_newest_sth()->set_tree_size(11);
  store2_->SetClusterNodeState(cns2);

  ClusterNodeState cns3;
  cns3.set_hostname(kNodeId3);
  cns3.set_log_port(9001);
  cns3.mutable_newest_sth()->set_timestamp(1002);
  cns3.mutable_newest_sth()->set_tree_size(9);
  store3_->SetClusterNodeState(cns3);
  sleep(1);

  // Have to choose cns3 (9@1002) here because we need 100% coverage:
  util::StatusOr<SignedTreeHead> sth1(controller50.GetCalculatedServingSTH());
  EXPECT_EQ(cns3.newest_sth().tree_size(), sth1.ValueOrDie().tree_size());
  EXPECT_EQ(cns3.newest_sth().timestamp(), sth1.ValueOrDie().timestamp());

  // Now cns3 moves to 13@1004
  cns3.mutable_newest_sth()->set_timestamp(1004);
  cns3.mutable_newest_sth()->set_tree_size(13);
  store3_->SetClusterNodeState(cns3);
  sleep(1);

  // Which means that the only STH from the current set that we can serve
  // must be 10@1002 (because coverage).
  // However, that timestamp was already used above, so the serving STH can't
  // have changed:
  util::StatusOr<SignedTreeHead> sth2(controller50.GetCalculatedServingSTH());
  EXPECT_EQ(sth1.ValueOrDie().DebugString(), sth2.ValueOrDie().DebugString());

  // Now cns1 moves to 13@1003
  cns3.mutable_newest_sth()->set_timestamp(1003);
  cns3.mutable_newest_sth()->set_tree_size(13);
  store3_->SetClusterNodeState(cns3);
  sleep(1);

  // Which means that the only STH from the current set that we can serve
  // must be 11@1000 (because coverage).
  // But that's in the past compared to Serving STH, so no dice.
  util::StatusOr<SignedTreeHead> sth3(controller50.GetCalculatedServingSTH());
  EXPECT_EQ(sth1.ValueOrDie().DebugString(), sth3.ValueOrDie().DebugString());

  // Finally cns2 moves to 13@1006
  cns2.mutable_newest_sth()->set_timestamp(1006);
  cns2.mutable_newest_sth()->set_tree_size(13);
  store2_->SetClusterNodeState(cns2);
  // And cns1 moves to 16@1003
  cns1.mutable_newest_sth()->set_timestamp(1006);
  cns1.mutable_newest_sth()->set_tree_size(13);
  store1_->SetClusterNodeState(cns1);
  sleep(1);

  // And we've got: 16@1002, 13@1006, 13@1003
  // So the cluster can move forward with its Serving STH
  util::StatusOr<SignedTreeHead> sth4(controller50.GetCalculatedServingSTH());
  EXPECT_EQ(cns2.newest_sth().tree_size(), sth4.ValueOrDie().tree_size());
  EXPECT_EQ(cns2.newest_sth().timestamp(), sth4.ValueOrDie().timestamp());
}


TEST_F(ClusterStateControllerTest,
       TestConfigChangesCauseServingSTHToBeRecalculated) {
  NiceMock<MockMasterElection> election_is_master;
  EXPECT_CALL(election_is_master, IsMaster()).WillRepeatedly(Return(true));
  ClusterStateController<LoggedEntry> controller(&pool_, base_, &url_fetcher_,
                                                 test_db_.db(), store1_.get(),
                                                 &election_is_master,
                                                 &fetcher_);
  SetClusterConfig(store1_.get(), 0 /* nodes */, 0.5 /* fraction */);
  store1_->SetClusterNodeState(cns100_);
  store2_->SetClusterNodeState(cns200_);
  store3_->SetClusterNodeState(cns300_);
  sleep(1);
  StatusOr<SignedTreeHead> sth(controller.GetCalculatedServingSTH());
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());

  SetClusterConfig(store1_.get(), 0 /* nodes */, 0.9 /* fraction */);
  sleep(1);
  sth = controller.GetCalculatedServingSTH();
  // You might expect sth100 here, but we shouldn't move to a smaller STH
  EXPECT_EQ(sth200_.tree_size(), sth.ValueOrDie().tree_size());

  SetClusterConfig(store1_.get(), 0 /* nodes */, 0.3 /* fraction */);
  sleep(1);
  sth = controller.GetCalculatedServingSTH();
  // Should be able to move to sth300 now.
  EXPECT_EQ(sth300_.tree_size(), sth.ValueOrDie().tree_size());
}


TEST_F(ClusterStateControllerTest, TestGetLocalNodeState) {
  SignedTreeHead sth;
  sth.set_timestamp(10000);
  sth.set_tree_size(2344);
  controller_.NewTreeHead(sth);

  ClusterNodeState state;
  controller_.GetLocalNodeState(&state);
  EXPECT_EQ(sth.DebugString(), state.newest_sth().DebugString());
}


TEST_F(ClusterStateControllerTest, TestNodeHostPort) {
  // Allow some time for our view of the node state to stabilize and then
  // check that it has. Ideally we would wait for this to definitely happen
  // but there seems to be no easy way to arrange this.
  sleep(1);
  const ClusterNodeState orig_node_state(GetNodeStateView(kNodeId1));
  EXPECT_EQ(kNodeId1, orig_node_state.hostname());
  EXPECT_EQ(9001, orig_node_state.log_port());

  // Now try to change the state and again allow some time for our view
  // to update
  const string kHost("myhostname");
  const int kPort(9999);

  controller_.SetNodeHostPort(kHost, kPort);
  sleep(1);

  const ClusterNodeState node_state(GetNodeStateView(kNodeId1));
  EXPECT_EQ(kHost, node_state.hostname());
  EXPECT_EQ(kPort, node_state.log_port());
}


TEST_F(ClusterStateControllerTest, TestStoresServingSthInDatabase) {
  SignedTreeHead sth;
  sth.set_timestamp(10000);
  sth.set_tree_size(0);
  store1_->SetServingSTH(sth);
  sleep(1);

  {
    SignedTreeHead db_sth;
    EXPECT_EQ(Database::LOOKUP_OK, test_db_.db()->LatestTreeHead(&db_sth));
    EXPECT_EQ(sth.DebugString(), db_sth.DebugString());
  }
}


TEST_F(ClusterStateControllerTest, TestWaitsToStoreSTHInDatabaseWhenStale) {
  SignedTreeHead sth1;
  sth1.set_timestamp(10000);
  sth1.set_tree_size(0);
  store1_->SetServingSTH(sth1);
  sleep(1);

  {
    SignedTreeHead db_sth;
    EXPECT_EQ(Database::LOOKUP_OK, test_db_.db()->LatestTreeHead(&db_sth));
    EXPECT_EQ(sth1.DebugString(), db_sth.DebugString());
  }

  SignedTreeHead sth2;
  sth2.set_timestamp(10001);
  sth2.set_tree_size(1);
  store1_->SetServingSTH(sth2);
  sleep(1);

  {
    // Should still show the first STH in sth1, because the local DB
    // doesn't have the entries under sth2 yet.
    SignedTreeHead db_sth;
    EXPECT_EQ(Database::LOOKUP_OK, test_db_.db()->LatestTreeHead(&db_sth));
    EXPECT_EQ(sth1.DebugString(), db_sth.DebugString());
  }

  // Now pretend the local fetcher has got lots of new entries from another
  // node, and the local signer has integrated them into our local tree:
  SignedTreeHead new_local_sth;
  new_local_sth.set_timestamp(10005);
  new_local_sth.set_tree_size(100);
  controller_.NewTreeHead(new_local_sth);

  {
    // Should now see the updated serving sth2
    SignedTreeHead db_sth;
    EXPECT_EQ(Database::LOOKUP_OK, test_db_.db()->LatestTreeHead(&db_sth));
    EXPECT_EQ(sth2.DebugString(), db_sth.DebugString());
  }
}


TEST_F(ClusterStateControllerTest, TestNodeIsStale) {
  EXPECT_TRUE(controller_.NodeIsStale());  // no STH yet.

  {
    SignedTreeHead sth;
    sth.set_timestamp(10000);
    sth.set_tree_size(0);
    store1_->SetServingSTH(sth);
    sleep(1);
  }

  EXPECT_FALSE(controller_.NodeIsStale());  // Have an STH we can serve.

  {
    SignedTreeHead sth;
    sth.set_timestamp(10001);
    sth.set_tree_size(1);
    store1_->SetServingSTH(sth);
    sleep(1);
  }

  EXPECT_TRUE(controller_.NodeIsStale());  // DB doesn't have the leaf

  {
    LoggedEntry cert;
    cert.RandomForTest();
    cert.set_sequence_number(0);
    EXPECT_EQ(Database::OK, test_db_.db()->CreateSequencedEntry(cert));
  }

  EXPECT_FALSE(controller_.NodeIsStale());
}


TEST_F(ClusterStateControllerTest, TestGetFreshNodes) {
  ClusterStateController<LoggedEntry> c2(&pool_, base_, &url_fetcher_,
                                         test_db_.db(), store2_.get(),
                                         &election2_, &fetcher_);
  ClusterStateController<LoggedEntry> c3(&pool_, base_, &url_fetcher_,
                                         test_db_.db(), store3_.get(),
                                         &election3_, &fetcher_);
  store1_->SetClusterNodeState(cns100_);
  store2_->SetClusterNodeState(cns200_);
  store3_->SetClusterNodeState(cns300_);

  {
    vector<ClusterNodeState> fresh(controller_.GetFreshNodes());
    EXPECT_EQ(static_cast<size_t>(0),
              fresh.size());  // no STH yet - everyone is stale.
  }

  // The 3 nodes have states which claim to have 100, 200, and 300 certs in
  // their DBs, iterate around setting the ServingSTH higher and higher to see
  // that they all go stale at the appropriate time.
  // This should knock out nodes based on store1_, store2_, store3_ in that
  // same order.
  const vector<vector<string>> kExpectedFreshNodes{
      {kNodeId2, kNodeId3},  // not including ourself (kNodeId1)
      {kNodeId2, kNodeId3},  // kNodeId1 is stale this time, so no change.
      {kNodeId3},
      {}};
  for (int i = 0; i < 4; ++i) {
    LOG(INFO) << "Iteration " << i;
    SignedTreeHead sth;
    sth.set_timestamp(i * 100 + 1);
    sth.set_tree_size(i * 100 + 1);
    store1_->SetServingSTH(sth);
    store2_->SetServingSTH(sth);
    store3_->SetServingSTH(sth);
    sleep(1);

    vector<string> ids;
    for (const auto& n : controller_.GetFreshNodes()) {
      ids.push_back(n.node_id());
    }
    EXPECT_EQ(ids, kExpectedFreshNodes[i]);
  }
}


}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  ConfigureSerializerForV1CT();
  return RUN_ALL_TESTS();
}
