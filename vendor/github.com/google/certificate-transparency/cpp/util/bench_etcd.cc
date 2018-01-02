#include <event2/thread.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "util/etcd.h"
#include "util/libevent_wrapper.h"
#include "util/sync_task.h"
#include "util/thread_pool.h"

namespace libevent = cert_trans::libevent;

using cert_trans::EtcdClient;
using cert_trans::ThreadPool;
using cert_trans::UrlFetcher;
using std::bind;
using std::make_shared;
using std::placeholders::_1;
using std::shared_ptr;
using std::string;
using std::thread;
using std::to_string;
using std::vector;
using util::Status;
using util::SyncTask;
using util::Task;

DEFINE_string(etcd, "127.0.0.1", "etcd server address");
DEFINE_int32(etcd_port, 4001, "etcd server port");
DEFINE_int32(requests_per_thread, 10, "number of requests per thread");
DEFINE_int32(bytes_per_request, 10, "number of bytes per requests");
DEFINE_int32(num_threads, 1, "number of threads");
DEFINE_string(test_key, "/bench_etcd", "base etcd key for testing");

namespace {


struct State {
  State(EtcdClient* etcd, int thread_num, Task* task)
      : etcd_(CHECK_NOTNULL(etcd)),
        key_prefix_(FLAGS_test_key + "/" + to_string(thread_num) + "/"),
        task_(CHECK_NOTNULL(task)),
        data_(FLAGS_bytes_per_request, 'x'),
        next_key_(0),
        num_left_(FLAGS_requests_per_thread) {
    CHECK_GT(num_left_, 0);
  }

  void MakeRequest();
  void RequestDone(Task* child_task);

  EtcdClient* const etcd_;
  const string key_prefix_;
  Task* const task_;
  const string data_;

  int64_t next_key_;
  EtcdClient::Response resp_;
  int num_left_;
};


void State::MakeRequest() {
  etcd_->Create(key_prefix_ + to_string(next_key_), "value", &resp_,
                task_->AddChild(bind(&State::RequestDone, this, _1)));
}


void State::RequestDone(Task* child_task) {
  CHECK_EQ(Status::OK, child_task->status());
  --num_left_;
  next_key_ = resp_.etcd_index;

  if (num_left_ > 0) {
    MakeRequest();
  } else {
    task_->Return();
  }
}


void test_etcd(int thread_num) {
  const shared_ptr<libevent::Base> event_base(make_shared<libevent::Base>());
  libevent::EventPumpThread pump(event_base);
  ThreadPool pool;
  UrlFetcher fetcher(event_base.get(), &pool);
  EtcdClient etcd(&pool, &fetcher, FLAGS_etcd, FLAGS_etcd_port);
  SyncTask task(event_base.get());
  State state(&etcd, thread_num, task.task());

  // Get the ball rolling...
  state.MakeRequest();

  LOG(INFO) << "waiting for test completion";
  task.Wait();
  LOG(INFO) << "test complete";
}


}  // namespace


int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  evthread_use_pthreads();

  CHECK_GT(FLAGS_requests_per_thread, 0);
  CHECK_GE(FLAGS_bytes_per_request, 0);
  CHECK_GT(FLAGS_num_threads, 0);

  vector<thread> threads;
  for (int i = 0; i < FLAGS_num_threads; ++i) {
    threads.emplace_back(bind(test_etcd, i));
  }

  for (vector<thread>::iterator it = threads.begin(); it != threads.end();
       ++it) {
    it->join();
  }

  return 0;
}
