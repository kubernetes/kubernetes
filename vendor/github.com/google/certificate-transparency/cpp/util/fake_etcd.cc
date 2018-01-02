#include "util/fake_etcd.h"

#include <glog/logging.h>

#include "util/json_wrapper.h"

using std::bind;
using std::chrono::seconds;
using std::chrono::system_clock;
using std::function;
using std::get;
using std::lock_guard;
using std::make_shared;
using std::make_tuple;
using std::map;
using std::move;
using std::multimap;
using std::mutex;
using std::ostringstream;
using std::shared_ptr;
using std::stoi;
using std::string;
using std::to_string;
using std::tuple;
using std::unique_lock;
using std::vector;
using util::Status;
using util::StatusOr;
using util::Task;

namespace cert_trans {
namespace {


string NormalizeKey(const string& input) {
  CHECK(!input.empty());
  CHECK_EQ(input[0], '/');

  string::size_type offset(0);
  string output;
  while (offset < input.size()) {
    const string::size_type next_part(input.find_first_not_of('/', offset));
    const string::size_type next_slash(input.find_first_of('/', next_part));

    if (next_part != string::npos) {
      const string part(
          input.substr(next_part - 1, next_slash - next_part + 1));

      if (part == "/..") {
        const string::size_type prev_slash(output.find_last_of('/'));
        if (prev_slash != string::npos) {
          output.erase(prev_slash);
        }
      } else if (part != "/.") {
        output.append(part);
      }
    }

    offset = next_slash;
  }

  if (output.empty()) {
    output.push_back('/');
  }

  return output;
}


bool WatchMatchesKey(const string& watch_key, const string& notify_key,
                     bool recursive) {
  if (watch_key == notify_key) {
    return true;
  }

  if (!recursive) {
    return false;
  }

  const string dir(watch_key + "/");
  return watch_key == "/" || notify_key.compare(0, dir.size(), dir) == 0;
}


}  // namespace

namespace {
const char* kStoreStats[] = {"setsFail",
                             "getsSuccess",
                             "watchers",
                             "expireCount",
                             "createFail",
                             "setsSuccess",
                             "compareAndDeleteFail",
                             "createSuccess",
                             "deleteFail",
                             "compareAndSwapSuccess",
                             "compareAndSwapFail",
                             "compareAndDeleteSuccess",
                             "updateFail",
                             "deleteSuccess",
                             "updateSuccess",
                             "getsFail"};
}  // namespace


FakeEtcdClient::FakeEtcdClient(libevent::Base* base)
    : base_(CHECK_NOTNULL(base)), parent_task_(base_), index_(1) {
  for (const auto& s : kStoreStats) {
    stats_[s] = 0;
  }
}


FakeEtcdClient::~FakeEtcdClient() {
  parent_task_.task()->Return();
  parent_task_.Wait();
  CHECK_EQ(parent_task_.status(), Status::OK);
}


void FakeEtcdClient::DumpEntries(const unique_lock<mutex>& lock) const {
  CHECK(lock.owns_lock());
  for (const auto& pair : entries_) {
    VLOG(1) << pair.second.ToString();
  }
}


void FakeEtcdClient::Watch(const string& rawkey, const WatchCallback& cb,
                           Task* task) {
  const string key(NormalizeKey(rawkey));
  unique_lock<mutex> lock(mutex_);
  vector<Node> initial_updates;
  map<string, Node>::const_iterator it(entries_.find(key));
  if (it != entries_.end()) {
    if (it->second.is_dir_) {
      const string key_prefix(key + "/");
      for (++it; it != entries_.end() &&
                 it->first.compare(0, key_prefix.size(), key_prefix) == 0;
           ++it) {
        CHECK(!it->second.deleted_);
        initial_updates.emplace_back(it->second);
      }
    } else {
      CHECK(!it->second.deleted_);
      initial_updates.emplace_back(it->second);
    }
  }
  ScheduleWatchCallback(lock, task, bind(cb, move(initial_updates)));
  watches_[key].push_back(make_pair(cb, task));
  task->WhenCancelled(bind(&FakeEtcdClient::CancelWatch, this, task));
  ++stats_["watchers"];
}


void FakeEtcdClient::PurgeExpiredEntriesWithLock(
    const unique_lock<mutex>& lock) {
  CHECK(lock.owns_lock());
  for (auto it = entries_.begin(); it != entries_.end();) {
    if (it->second.expires_ < system_clock::now()) {
      VLOG(1) << "Deleting expired entry " << it->first;
      it->second.deleted_ = true;
      NotifyForPath(lock, it->first);
      it = entries_.erase(it);
      ++stats_["expireCount"];
    } else {
      ++it;
    }
  }
}


void FakeEtcdClient::PurgeExpiredEntries() {
  unique_lock<mutex> lock(mutex_);
  PurgeExpiredEntriesWithLock(lock);
}


void FakeEtcdClient::NotifyForPath(const unique_lock<mutex>& lock,
                                   const string& path) {
  CHECK(lock.owns_lock());
  VLOG(1) << "notifying " << path;
  const map<string, Node>::const_iterator node_it(entries_.find(path));
  CHECK(node_it != entries_.end());
  const Node& node(node_it->second);

  const multimap<string, tuple<bool, GetResponse*, Task*>>::iterator last(
      waiting_gets_.upper_bound(path));
  multimap<string, tuple<bool, GetResponse*, Task*>>::iterator it;
  for (it = waiting_gets_.begin(); it != last;) {
    if (WatchMatchesKey(it->first, node.key_, get<0>(it->second))) {
      get<1>(it->second)->node = node;
      get<2>(it->second)->Return();
      it = waiting_gets_.erase(it);
    } else {
      ++it;
    }
  }

  for (const auto& pair : watches_) {
    if (path.find(pair.first) == 0) {
      for (const auto& cb_cookie : pair.second) {
        ScheduleWatchCallback(lock, cb_cookie.second,
                              bind(cb_cookie.first, vector<Node>{node}));
      }
    }
  }
}


void FakeEtcdClient::Get(const Request& req, GetResponse* resp, Task* task) {
  VLOG(1) << "GET " << req.key;
  const string key(NormalizeKey(req.key));

  CHECK_NE(key, "/") << "not implemented";

  task->CleanupWhenDone(
      bind(&FakeEtcdClient::UpdateOperationStats, this, "gets", task));

  unique_lock<mutex> lock(mutex_);
  PurgeExpiredEntriesWithLock(lock);
  resp->etcd_index = index_;

  if (req.wait_index > 0) {
    if (req.wait_index <= index_) {
      // Our fake history log is *very* small!
      task->Return(
          Status(util::error::ABORTED,
                 "The event in requested index is outdated and cleared"));
      return;
    }

    waiting_gets_.insert(
        make_pair(key, make_tuple(req.recursive, resp, task)));
    task->WhenCancelled(
        bind(&FakeEtcdClient::CancelWaitingGet, this, key, task));
    return;
  }

  map<string, Node>::const_iterator it(entries_.find(key));
  if (it == entries_.end()) {
    task->Return(Status(util::error::NOT_FOUND, "not found"));
    return;
  }
  resp->node = it->second;
  ++it;

  vector<Node*> parent_nodes{&resp->node};
  while (!parent_nodes.empty() && it != entries_.end()) {
    const string key_prefix(parent_nodes.back()->key_ + "/");
    if (it->first.compare(0, key_prefix.size(), key_prefix) > 0) {
      parent_nodes.pop_back();
      continue;
    }

    if (req.recursive) {
      parent_nodes.back()->nodes_.emplace_back(it->second);

      if (it->second.is_dir_) {
        // The node we just added is now the current parent node.
        parent_nodes.push_back(&parent_nodes.back()->nodes_.back());
      }
    } else {
      if (it->first.find_first_of('/', key_prefix.size()) == string::npos) {
        resp->node.nodes_.emplace_back(it->second);
      }
    }

    ++it;
  }
  task->Return();
}


void FakeEtcdClient::InternalPut(const string& rawkey, const string& value,
                                 const system_clock::time_point& expires,
                                 bool create, int64_t prev_index,
                                 Response* resp, Task* task) {
  const string key(NormalizeKey(rawkey));
  CHECK_NE(key.back(), '/');
  CHECK(!create || prev_index <= 0);

  vector<string> parents;
  for (string::size_type offset = 0; offset < key.size();) {
    const string::size_type next_slash(key.find_first_of('/', offset + 1));
    if (next_slash == string::npos) {
      break;
    }

    parents.emplace_back(key.substr(offset + 1, next_slash - offset - 1));
    offset = next_slash;
  }

  *resp = EtcdClient::Response();
  unique_lock<mutex> lock(mutex_);
  PurgeExpiredEntriesWithLock(lock);
  const int64_t new_index(index_ + 1);

  // If we're creating, make sure all the parent entries exist and are
  // directories, creating them as needed.
  if (create) {
    string parent_key;
    for (const string& path : parents) {
      parent_key.append("/" + path);
      // Either this inserts a directory, or it fails, and it gets us
      // the entry. If we insert a directory, then success is
      // guaranteed below, so we're sure to update index_
      // appropriately.
      if (!entries_.insert(make_pair(parent_key,
                                     Node(new_index, new_index, parent_key,
                                          true, "", {}, false)))
               .first->second.is_dir_) {
        task->Return(Status(util::error::ABORTED, "Not a directory"));
        return;
      }
    }
  }

  Node node(new_index, new_index, key, false, value, {}, false);
  node.expires_ = expires;
  const map<string, Node>::const_iterator entry(entries_.find(key));
  if (create && entry != entries_.end()) {
    task->Return(
        Status(util::error::FAILED_PRECONDITION, key + " already exists"));
    return;
  }

  if (prev_index > 0) {
    if (entry == entries_.end()) {
      task->Return(Status(util::error::FAILED_PRECONDITION,
                          "node doesn't exist: " + key));
      return;
    }
    if (prev_index != entry->second.modified_index_) {
      task->Return(Status(util::error::FAILED_PRECONDITION,
                          "incorrect index:  prevIndex=" +
                              to_string(prev_index) + " but modified_index_=" +
                              to_string(entry->second.modified_index_)));
      return;
    }
    node.created_index_ = entry->second.created_index_;
  }

  entries_[key] = node;
  resp->etcd_index = new_index;
  index_ = new_index;
  task->Return();
  NotifyForPath(lock, key);
  DumpEntries(lock);
  if (expires < system_clock::time_point::max()) {
    const std::chrono::duration<double> delay(expires - system_clock::now());
    base_->Delay(delay, parent_task_.task()->AddChild(
                            bind(&FakeEtcdClient::PurgeExpiredEntries, this)));
  }
}


void FakeEtcdClient::InternalDelete(const string& key,
                                    const int64_t current_index, Task* task) {
  VLOG(1) << "DELETE " << key;
  CHECK(!key.empty());
  CHECK_EQ(key.front(), '/');
  CHECK_NE(key.back(), '/');

  const string op_name(current_index > 0 ? "compareAndDelete" : "delete");

  unique_lock<mutex> lock(mutex_);
  PurgeExpiredEntriesWithLock(lock);
  const map<string, Node>::iterator entry(entries_.find(key));
  if (entry == entries_.end()) {
    ++stats_[op_name + "Fail"];
    task->Return(Status(util::error::NOT_FOUND, "Node doesn't exist: " + key));
    return;
  }
  if (current_index > 0 && entry->second.modified_index_ != current_index) {
    ++stats_[op_name + "Fail"];
    task->Return(Status(util::error::FAILED_PRECONDITION,
                        "Incorrect index:  prevIndex=" +
                            to_string(current_index) +
                            " but modified_index_=" +
                            to_string(entry->second.modified_index_)));
    return;
  }
  entry->second.modified_index_ = ++index_;
  entry->second.value_.clear();
  entry->second.deleted_ = true;
  ++stats_[op_name + "Success"];
  task->Return();
  NotifyForPath(lock, key);
  entries_.erase(entry);
}


void FakeEtcdClient::UpdateOperationStats(const string& op, const Task* task) {
  CHECK_NOTNULL(task);
  if (!task->IsActive()) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_[op + (task->status().ok() ? "Success" : "Fail")];
  }
}


void FakeEtcdClient::Create(const string& key, const string& value,
                            Response* resp, Task* task) {
  task->CleanupWhenDone(
      bind(&FakeEtcdClient::UpdateOperationStats, this, "create", task));
  InternalPut(key, value, system_clock::time_point::max(), true, -1, resp,
              task);
}


void FakeEtcdClient::CreateWithTTL(const string& key, const string& value,
                                   const seconds& ttl, Response* resp,
                                   Task* task) {
  task->CleanupWhenDone(
      bind(&FakeEtcdClient::UpdateOperationStats, this, "create", task));
  InternalPut(key, value, system_clock::now() + ttl, true, -1, resp, task);
}


void FakeEtcdClient::Update(const string& key, const string& value,
                            const int64_t previous_index, Response* resp,
                            Task* task) {
  task->CleanupWhenDone(bind(&FakeEtcdClient::UpdateOperationStats, this,
                             "compareAndSwap", task));
  InternalPut(key, value, system_clock::time_point::max(), false,
              previous_index, resp, task);
}


void FakeEtcdClient::UpdateWithTTL(const string& key, const string& value,
                                   const seconds& ttl,
                                   const int64_t previous_index,
                                   Response* resp, Task* task) {
  task->CleanupWhenDone(bind(&FakeEtcdClient::UpdateOperationStats, this,
                             "compareAndSwap", task));
  InternalPut(key, value, system_clock::now() + ttl, false, previous_index,
              resp, task);
}


void FakeEtcdClient::ForceSet(const string& key, const string& value,
                              Response* resp, Task* task) {
  task->CleanupWhenDone(
      bind(&FakeEtcdClient::UpdateOperationStats, this, "sets", task));
  InternalPut(key, value, system_clock::time_point::max(), false, -1, resp,
              task);
}


void FakeEtcdClient::ForceSetWithTTL(const std::string& key,
                                     const std::string& value,
                                     const std::chrono::seconds& ttl,
                                     Response* resp, util::Task* task) {
  task->CleanupWhenDone(
      bind(&FakeEtcdClient::UpdateOperationStats, this, "sets", task));
  InternalPut(key, value, system_clock::now() + ttl, false, -1, resp, task);
}


void FakeEtcdClient::Delete(const string& key, const int64_t current_index,
                            Task* task) {
  CHECK_GT(current_index, 0);
  InternalDelete(key, current_index, task);
}


void FakeEtcdClient::ForceDelete(const string& key, Task* task) {
  InternalDelete(key, 0, task);
}


void FakeEtcdClient::GetStoreStats(StatsResponse* resp, Task* task) {
  CHECK_NOTNULL(resp);
  CHECK_NOTNULL(task);
  resp->stats = stats_;
  task->Return();
}


void FakeEtcdClient::CancelWatch(Task* task) {
  lock_guard<mutex> lock(mutex_);
  bool found(false);
  for (auto& pair : watches_) {
    for (auto it(pair.second.begin()); it != pair.second.end();) {
      if (it->second == task) {
        CHECK(!found);
        found = true;
        VLOG(1) << "Removing watcher " << it->second << " on " << pair.first;
        --stats_["watchers"];
        // Outstanding notifications have a hold on this task, so they
        // will all go through before the task actually completes. But
        // we won't be sending new notifications.
        task->Return(Status::CANCELLED);
        it = pair.second.erase(it);
      } else {
        ++it;
      }
    }
  }
}


void FakeEtcdClient::CancelWaitingGet(const string& key, Task* task) {
  lock_guard<mutex> lock(mutex_);
  const multimap<string, tuple<bool, GetResponse*, Task*>>::iterator last(
      waiting_gets_.upper_bound(key));

  for (auto it = waiting_gets_.lower_bound(key); it != last; ++it) {
    if (get<2>(it->second) == task) {
      waiting_gets_.erase(it);
      task->Return(Status::CANCELLED);
      return;
    }
  }
}


void FakeEtcdClient::ScheduleWatchCallback(
    const unique_lock<mutex>& lock, Task* task,
    const std::function<void()>& callback) {
  CHECK(lock.owns_lock());
  const bool already_running(!watches_callbacks_.empty());

  task->AddHold();
  watches_callbacks_.emplace_back(make_pair(task, move(callback)));

  // TODO(pphaneuf): This might fare poorly if the executor is
  // synchronous.
  if (!already_running) {
    watches_callbacks_.front().first->executor()->Add(
        bind(&FakeEtcdClient::RunWatchCallback, this));
  }
}


void FakeEtcdClient::RunWatchCallback() {
  Task* current(nullptr);
  Task* next(nullptr);
  function<void()> callback;

  {
    lock_guard<mutex> lock(mutex_);

    CHECK(!watches_callbacks_.empty());
    current = move(watches_callbacks_.front().first);
    callback = move(watches_callbacks_.front().second);
    watches_callbacks_.pop_front();

    if (!watches_callbacks_.empty()) {
      next = CHECK_NOTNULL(watches_callbacks_.front().first);
    }
  }

  callback();
  current->RemoveHold();

  // If we have a next executor, schedule ourselves on it.
  if (next) {
    next->executor()->Add(bind(&FakeEtcdClient::RunWatchCallback, this));
  }
}


}  // namespace cert_trans
