#include "util/masterelection.h"

#include <event2/thread.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <atomic>
#include <map>
#include <string>

#include "base/notification.h"
#include "util/etcd.h"
#include "util/fake_etcd.h"
#include "util/periodic_closure.h"
#include "util/status_test_util.h"
#include "util/testing.h"
#include "util/thread_pool.h"

DECLARE_string(trusted_root_certs);
DEFINE_string(cert_dir, "test/testdata/urlfetcher_test_certs",
              "Directory containing the test certs.");

namespace cert_trans {

using cert_trans::Notification;
using std::atomic;
using std::bind;
using std::chrono::seconds;
using std::function;
using std::make_shared;
using std::map;
using std::placeholders::_1;
using std::placeholders::_2;
using std::shared_ptr;
using std::string;
using std::thread;
using std::to_string;
using std::unique_ptr;
using std::vector;
using testing::AllOf;
using testing::Contains;
using testing::InvokeArgument;
using testing::Pair;
using testing::_;
using util::Status;
using util::SyncTask;


const char kProposalDir[] = "/master/";

DEFINE_string(etcd, "", "etcd server address");
DEFINE_int32(etcd_port, 4001, "etcd server port");
DECLARE_int32(master_keepalive_interval_seconds);
DECLARE_int32(masterelection_retry_delay_seconds);


// Simple helper class, represents a thread of interest in participating in
// an election.
struct Participant {
  // Constructs a new MasterElection object, and immediately starts
  // participating in the election.
  Participant(const string& dir, const string& id,
              const shared_ptr<libevent::Base>& base, EtcdClient* client)
      : base_(base),
        client_(CHECK_NOTNULL(client)),
        election_(new MasterElection(base_, client_, dir, id)),
        dir_(dir),
        id_(id),
        mastership_count_(0) {
    EXPECT_FALSE(election_->IsMaster()) << id_;
  }

  void StartElection() {
    election_->StartElection();
  }

  void StopElection() {
    VLOG(1) << id_ << " about to StopElection().";
    election_->StopElection();
    VLOG(1) << id_ << " completed StopElection().";
    EXPECT_FALSE(election_->IsMaster()) << id_;
  }

  // Wait to become the boss!
  void ElectLikeABoss() {
    StartElection();
    VLOG(1) << id_ << " about to WaitToBecomeMaster().";
    election_->WaitToBecomeMaster();
    EXPECT_TRUE(election_->IsMaster()) << id_;
    ++mastership_count_;
    VLOG(1) << id_ << " completed WaitToBecomeMaster().";
  }

  bool WaitToBecomeMaster() {
    return election_->WaitToBecomeMaster();
  }

  bool IsMaster() {
    return election_->IsMaster();
  }

  void ElectionMania(int num_rounds,
                     const vector<unique_ptr<Participant>>* all_participants) {
    notification_.reset(new Notification);
    mania_thread_.reset(new thread([this, num_rounds, all_participants]() {
      for (int round(0); round < num_rounds; ++round) {
        VLOG(1) << id_ << " starting round " << round;
        ElectLikeABoss();

        int num_masters(0);
        for (const auto& participant : *all_participants) {
          if (participant->election_->IsMaster()) {
            ++num_masters;
          }
        }
        // There /could/ be no masters if an update happened after we came
        // out of WaitToBecomeMaster, it's unlikely but possible.
        // There definitely shouldn't be > 1 master EVER, though.
        CHECK_LE(num_masters, 1) << "From the PoV of " << id_;
        StopElection();
        VLOG(1) << id_ << " finished round " << round;
        // Restarting an existing MasterElection and creating a new
        // one should both work, so test both cases on various rounds.
        if ((round % 2) == 0) {
          election_.reset(new MasterElection(base_, client_, dir_, id_));
        }
      }
      VLOG(1) << id_ << " Mania over!";
      notification_->Notify();
    }));
  }

  void WaitForManiaToEnd() {
    CHECK(notification_);
    notification_->WaitForNotification();
    mania_thread_->join();
    mania_thread_.reset();
  }

  const shared_ptr<libevent::Base>& base_;
  EtcdClient* const client_;
  unique_ptr<MasterElection> election_;
  unique_ptr<Notification> notification_;
  unique_ptr<thread> mania_thread_;
  const string dir_;
  const string id_;
  atomic<int> mastership_count_;
};


class ElectionTest : public ::testing::Test {
 public:
  ElectionTest()
      : base_(make_shared<libevent::Base>()),
        event_pump_(base_),
        pool_(),
        url_fetcher_(base_.get(), &pool_),
        client_(FLAGS_etcd.empty()
                    ? new FakeEtcdClient(base_.get())
                    : new EtcdClient(&pool_, &url_fetcher_, FLAGS_etcd,
                                     FLAGS_etcd_port)) {
  }


 protected:
  void KillProposalRefresh(Participant* p) {
    p->election_->proposal_refresh_callback_.reset();
  }


  shared_ptr<libevent::Base> base_;
  libevent::EventPumpThread event_pump_;
  ThreadPool pool_;
  UrlFetcher url_fetcher_;
  atomic<bool> running_;
  const unique_ptr<EtcdClient> client_;
};


typedef class ElectionTest ElectionDeathTest;


TEST_F(ElectionTest, SingleInstanceBecomesMaster) {
  Participant one(kProposalDir, "1", base_, client_.get());
  EXPECT_FALSE(one.IsMaster());

  one.ElectLikeABoss();
  EXPECT_TRUE(one.IsMaster());

  one.StopElection();
  EXPECT_FALSE(one.IsMaster());
}


TEST_F(ElectionTest, MultiInstanceElection) {
  Participant one(kProposalDir, "1", base_, client_.get());
  one.ElectLikeABoss();
  EXPECT_TRUE(one.IsMaster());

  Participant two(kProposalDir, "2", base_, client_.get());
  two.StartElection();
  sleep(1);
  EXPECT_FALSE(two.IsMaster());

  Participant three(kProposalDir, "3", base_, client_.get());
  three.StartElection();
  sleep(1);
  EXPECT_FALSE(three.IsMaster());

  EXPECT_TRUE(one.IsMaster());

  one.StopElection();
  EXPECT_FALSE(one.IsMaster());

  EXPECT_TRUE(two.WaitToBecomeMaster());
  EXPECT_FALSE(one.IsMaster());
  EXPECT_TRUE(two.IsMaster());
  EXPECT_FALSE(three.IsMaster());

  two.StopElection();
  EXPECT_FALSE(two.IsMaster());

  EXPECT_TRUE(three.WaitToBecomeMaster());
  EXPECT_FALSE(one.IsMaster());
  EXPECT_FALSE(two.IsMaster());
  EXPECT_TRUE(three.IsMaster());

  three.StopElection();
  EXPECT_FALSE(three.IsMaster());

  sleep(2);
  EXPECT_FALSE(one.IsMaster());
  EXPECT_FALSE(two.IsMaster());
  EXPECT_FALSE(three.IsMaster());
}


TEST_F(ElectionTest, RejoinElection) {
  Participant one(kProposalDir, "1", base_, client_.get());
  EXPECT_FALSE(one.IsMaster());

  one.ElectLikeABoss();
  EXPECT_TRUE(one.IsMaster());

  one.StopElection();
  EXPECT_FALSE(one.IsMaster());

  // Join in again:
  one.ElectLikeABoss();
  EXPECT_TRUE(one.IsMaster());

  one.StopElection();
  EXPECT_FALSE(one.IsMaster());
}


TEST_F(ElectionTest, OkToCallStartAndStopElectionMultipleTimes) {
  Participant one(kProposalDir, "1", base_, client_.get());
  one.StartElection();
  one.WaitToBecomeMaster();
  EXPECT_TRUE(one.IsMaster());
  one.StartElection();
  EXPECT_TRUE(one.IsMaster());

  one.StopElection();
  EXPECT_FALSE(one.IsMaster());
  one.StopElection();
  EXPECT_FALSE(one.IsMaster());
}


TEST_F(ElectionTest, RetresCreatingProposal) {
  FLAGS_masterelection_retry_delay_seconds = 1;
  {
    EtcdClient::Response resp;
    SyncTask task(base_.get());
    client_->Create(string(kProposalDir) + "1", "", &resp, task.task());
    task.Wait();
    ASSERT_OK(task.status());
  }

  Participant one(kProposalDir, "1", base_, client_.get());
  one.StartElection();
  sleep(2);
  EXPECT_FALSE(one.IsMaster());

  {
    SyncTask task(base_.get());
    client_->ForceDelete(string(kProposalDir) + "1", task.task());
    task.Wait();
    ASSERT_OK(task.status());
  }

  sleep(2);

  EXPECT_TRUE(one.IsMaster());
  one.StopElection();
}


TEST_F(ElectionTest, ElectionMania) {
  const int kNumRounds(20);
  const int kNumParticipants(20);
  vector<unique_ptr<Participant>> participants;
  participants.reserve(kNumParticipants);
  for (int i = 0; i < kNumParticipants; ++i) {
    participants.emplace_back(
        new Participant(kProposalDir, to_string(i), base_, client_.get()));
  };

  for (int i = 0; i < kNumParticipants; ++i) {
    participants[i]->ElectionMania(kNumRounds, &participants);
  }

  for (int i = 0; i < kNumParticipants; ++i) {
    LOG(INFO) << i << " became master " << participants[i]->mastership_count_
              << " times.";
    participants[i]->WaitForManiaToEnd();
  }
}


}  // namespace cert_trans


int main(int argc, char** argv) {
  cert_trans::test::InitTesting(argv[0], &argc, &argv, true);
  // Default value of trusted root certs may not be correct on all platforms
  FLAGS_trusted_root_certs = FLAGS_cert_dir + "/ca-cert.pem";
  return RUN_ALL_TESTS();
}
