#ifndef CERT_TRANS_UTIL_FAKE_ETCD_H_
#define CERT_TRANS_UTIL_FAKE_ETCD_H_

#include <deque>
#include <map>
#include <queue>
#include <string>
#include <tuple>

#include "util/etcd.h"
#include "util/libevent_wrapper.h"
#include "util/statusor.h"
#include "util/sync_task.h"
#include "util/task.h"

namespace cert_trans {


class FakeEtcdClient : public EtcdClient {
 public:
  explicit FakeEtcdClient(libevent::Base* base);

  virtual ~FakeEtcdClient();

  void Get(const Request& req, GetResponse* resp, util::Task* task) override;

  void Create(const std::string& key, const std::string& value, Response* resp,
              util::Task* task) override;

  void CreateWithTTL(const std::string& key, const std::string& value,
                     const std::chrono::seconds& ttl, Response* resp,
                     util::Task* task) override;

  void Update(const std::string& key, const std::string& value,
              const int64_t previous_index, Response* resp,
              util::Task* task) override;

  void UpdateWithTTL(const std::string& key, const std::string& value,
                     const std::chrono::seconds& ttl,
                     const int64_t previous_index, Response* resp,
                     util::Task* task) override;

  void ForceSet(const std::string& key, const std::string& value,
                Response* resp, util::Task* task) override;

  void ForceSetWithTTL(const std::string& key, const std::string& value,
                       const std::chrono::seconds& ttl, Response* resp,
                       util::Task* task) override;

  void Delete(const std::string& key, const int64_t current_index,
              util::Task* task) override;

  void ForceDelete(const std::string& key, util::Task* task) override;

  void GetStoreStats(StatsResponse* resp, util::Task* task) override;

  // The callbacks for *all* watches will be called one at a time, in
  // order, which is a stronger guarantee than the one
  // EtcdClient::Watch has.
  void Watch(const std::string& rawkey, const WatchCallback& cb,
             util::Task* task) override;

 private:
  void DumpEntries(const std::unique_lock<std::mutex>& lock) const;

  void PurgeExpiredEntriesWithLock(const std::unique_lock<std::mutex>& lock);
  void PurgeExpiredEntries();

  void NotifyForPath(const std::unique_lock<std::mutex>& lock,
                     const std::string& path);

  void InternalPut(const std::string& rawkey, const std::string& value,
                   const std::chrono::system_clock::time_point& expires,
                   bool create, int64_t prev_index, Response* resp,
                   util::Task* task);

  void InternalDelete(const std::string& key, const int64_t current_index,
                      util::Task* task);

  void UpdateOperationStats(const std::string& op, const util::Task* task);

  void CancelWatch(util::Task* task);
  void CancelWaitingGet(const std::string& key, util::Task* task);

  // Arranges for the watch callbacks to be called in order. Should be
  // called with mutex_ held.
  void ScheduleWatchCallback(const std::unique_lock<std::mutex>& lock,
                             util::Task* task,
                             const std::function<void()>& callback);
  void RunWatchCallback();

  libevent::Base* const base_;
  util::SyncTask parent_task_;
  std::mutex mutex_;
  int64_t index_;
  std::map<std::string, Node> entries_;
  std::multimap<std::string, std::tuple<bool, GetResponse*, util::Task*>>
      waiting_gets_;
  std::map<std::string, std::vector<std::pair<WatchCallback, util::Task*>>>
      watches_;
  std::deque<std::pair<util::Task*, std::function<void()>>> watches_callbacks_;
  std::map<std::string, int64_t> stats_;

  friend class ElectionTest;
};


}  // namespace cert_trans

#endif  // CERT_TRANS_UTIL_FAKE_ETCD_H_
