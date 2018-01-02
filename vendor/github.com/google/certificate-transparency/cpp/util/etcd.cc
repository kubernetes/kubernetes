#include "util/etcd.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ctime>
#include <utility>
#include <event2/http.h>

#include "util/json_wrapper.h"
#include "util/libevent_wrapper.h"
#include "util/statusor.h"

namespace libevent = cert_trans::libevent;

using std::atoll;
using std::bind;
using std::chrono::seconds;
using std::chrono::system_clock;
using std::ctime;
using std::list;
using std::lock_guard;
using std::make_pair;
using std::make_shared;
using std::map;
using std::max;
using std::move;
using std::mutex;
using std::ostringstream;
using std::placeholders::_1;
using std::shared_ptr;
using std::stoi;
using std::string;
using std::time_t;
using std::to_string;
using std::unique_ptr;
using std::vector;
using util::Executor;
using util::Status;
using util::StatusOr;
using util::SyncTask;
using util::Task;

DEFINE_int32(etcd_watch_error_retry_delay_seconds, 5,
             "delay between retrying etcd watch requests");
DEFINE_bool(etcd_consistent, true,
            "Add consistent=true param to all requests. Do not turn this off "
            "unless you *know* what you're doing.");
DEFINE_bool(etcd_quorum, true,
            "Add quorum=true param to all requests. Do not turn this off "
            "unless you *know* what you're doing.");
DEFINE_int32(etcd_connection_timeout_seconds, 10,
             "Number of seconds after which to timeout etcd connections.");

namespace cert_trans {

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

const char kKeysSpace[] = "/v2/keys";
const char kStatsSpace[] = "/v2/stats";

const char kStoreStatsKey[] = "/store";


util::error::Code ErrorCodeForHttpResponseCode(int response_code) {
  switch (response_code) {
    case 200:
    case 201:
      return util::error::OK;
    case 400:
      return util::error::ABORTED;
    case 403:
      return util::error::PERMISSION_DENIED;
    case 404:
      return util::error::NOT_FOUND;
    case 412:
      return util::error::FAILED_PRECONDITION;
    case 500:
      return util::error::UNAVAILABLE;
    default:
      return util::error::UNKNOWN;
  }
}


util::error::Code StatusCodeFromEtcdErrorCode(int etcd_code) {
  switch (etcd_code) {
    case 100:  // Key not found
      return util::error::NOT_FOUND;
    case 101:  // Compare failed
    case 102:  // Not a file
    case 104:  // Not a directory
    case 105:  // Key already exists
    case 107:  // Root is read-only
    case 108:  // Directory not empty
      return util::error::FAILED_PRECONDITION;

    case 201:  // PrevValue missing
    case 202:  // Provided TTL is not a number
    case 203:  // Provided index is not a number
    case 209:  // Invalid field
    case 210:  // Invalid POST form
      return util::error::FAILED_PRECONDITION;

    case 300:  // Raft Internal Error
    case 301:  // During Leader Election
      return util::error::UNAVAILABLE;

    case 400:  // Watcher is cleared due to etcd recovery
    case 401:  // Event in requested index is outdated and cleared.
      return util::error::ABORTED;

    default:
      return util::error::UNKNOWN;
  }
}


string EtcdErrorMessage(const JsonObject& json) {
  const JsonString message(json, "message");
  const JsonString cause(json, "cause");

  string ret("Etcd message: ");
  if (message.Ok()) {
    ret += message.Value();
    if (cause.Ok()) {
      ret += ", cause: " + string(cause.Value());
    }
  } else {
    ret = json.DebugString();
  }
  return ret;
}


Status StatusFromResponse(int response_code, const JsonObject& json) {
  // Prefer the etcd errorCode if there is one:
  if (json.Ok()) {
    const JsonInt error_code(json, "errorCode");
    if (error_code.Ok()) {
      return Status(StatusCodeFromEtcdErrorCode(error_code.Value()),
                    EtcdErrorMessage(json));
    }
  }
  // Otherwise use the HTTP code:
  const util::error::Code error_code(
      ErrorCodeForHttpResponseCode(response_code));
  const string error_message(
      error_code == util::error::OK ? "" : json.DebugString());
  return Status(error_code, error_message);
}


StatusOr<EtcdClient::Node> ParseNodeFromJson(const JsonObject& json_node) {
  const JsonInt createdIndex(json_node, "createdIndex");
  if (!createdIndex.Ok()) {
    return Status(util::error::FAILED_PRECONDITION,
                  "Invalid JSON: Couldn't find 'createdIndex'");
  }

  const JsonInt modifiedIndex(json_node, "modifiedIndex");
  if (!modifiedIndex.Ok()) {
    return Status(util::error::FAILED_PRECONDITION,
                  "Invalid JSON: Couldn't find 'modifiedIndex'");
  }

  const JsonString key(json_node, "key");
  if (!key.Ok()) {
    return Status(util::error::FAILED_PRECONDITION,
                  "Invalid JSON: Couldn't find 'key'");
  }

  const JsonString value(json_node, "value");
  const JsonBoolean isDir(json_node, "dir");
  const bool is_dir(isDir.Ok() && isDir.Value());
  const bool deleted(!value.Ok() && !is_dir);
  vector<EtcdClient::Node> nodes;
  if (is_dir && !deleted) {
    const JsonArray json_nodes(json_node, "nodes");
    if (json_nodes.Ok()) {
      for (int i = 0; i < json_nodes.Length(); ++i) {
        const JsonObject json_entry(json_nodes, i);
        if (!json_entry.Ok()) {
          return Status(util::error::FAILED_PRECONDITION,
                        "Invalid JSON: Couldn't get 'nodes' index " +
                            to_string(i));
        }

        StatusOr<EtcdClient::Node> entry(ParseNodeFromJson(json_entry));
        if (!entry.status().ok()) {
          return entry.status();
        }

        if (entry.ValueOrDie().deleted_) {
          return Status(util::error::FAILED_PRECONDITION,
                        "Deleted sub-node " + string(key.Value()));
        }

        nodes.emplace_back(entry.ValueOrDie());
      }
    }
  }

  return EtcdClient::Node(createdIndex.Value(), modifiedIndex.Value(),
                          key.Value(), is_dir,
                          (deleted || is_dir) ? "" : value.Value(),
                          move(nodes), deleted);
}


void GetRequestDone(const string& keyname, EtcdClient::GetResponse* resp,
                    Task* parent_task, EtcdClient::GenericResponse* gen_resp,
                    Task* task) {
  *resp = EtcdClient::GetResponse();
  resp->etcd_index = gen_resp->etcd_index;
  if (!task->status().ok()) {
    // TODO(pphaneuf): Handle connection timeout (status DEADLINE_EXCEEDED)
    // better here? Or add deadline support here and in UrlFetcher,
    // with retries, so that this doesn't get all the way here?
    parent_task->Return(
        Status(task->status().CanonicalCode(),
               task->status().error_message() + " (" + keyname + ")"));
    return;
  }

  const JsonObject json_node(*gen_resp->json_body, "node");
  if (!json_node.Ok()) {
    parent_task->Return(Status(util::error::FAILED_PRECONDITION,
                               "Invalid JSON: Couldn't find 'node'"));
    return;
  }

  StatusOr<EtcdClient::Node> node(ParseNodeFromJson(json_node));
  if (!node.status().ok()) {
    parent_task->Return(node.status());
    return;
  }

  resp->node = node.ValueOrDie();
  parent_task->Return();
}


void CopyStat(const string& key, const JsonObject& from,
              map<string, int64_t>* to) {
  CHECK_NOTNULL(to);
  const JsonInt stat(from, key.c_str());
  if (!stat.Ok()) {
    LOG(WARNING) << "Failed to find stat " << key;
    return;
  }
  (*to)[key] = stat.Value();
}


void GetStoreStatsRequestDone(EtcdClient::StatsResponse* resp,
                              Task* parent_task,
                              EtcdClient::GenericResponse* gen_resp,
                              Task* task) {
  *resp = EtcdClient::StatsResponse();
  resp->etcd_index = gen_resp->etcd_index;
  if (!task->status().ok()) {
    parent_task->Return(task->status());
    return;
  }

  if (!gen_resp->json_body->Ok()) {
    parent_task->Return(Status(util::error::FAILED_PRECONDITION,
                               "Invalid JSON: json_body not Ok."));
    return;
  }

  for (const auto& stat : kStoreStats) {
    CopyStat(stat, *gen_resp->json_body, &resp->stats);
  }
  parent_task->Return();
}


void CreateRequestDone(EtcdClient::Response* resp, Task* parent_task,
                       EtcdClient::GenericResponse* gen_resp, Task* task) {
  if (!task->status().ok()) {
    parent_task->Return(task->status());
    return;
  }

  const JsonObject json_node(*gen_resp->json_body, "node");
  if (!json_node.Ok()) {
    parent_task->Return(Status(util::error::FAILED_PRECONDITION,
                               "Invalid JSON: Couldn't find 'node'"));
    return;
  }

  StatusOr<EtcdClient::Node> node(ParseNodeFromJson(json_node));
  if (!node.status().ok()) {
    parent_task->Return(node.status());
    return;
  }

  CHECK_EQ(node.ValueOrDie().created_index_,
           node.ValueOrDie().modified_index_);
  resp->etcd_index = node.ValueOrDie().modified_index_;
  parent_task->Return();
}


void UpdateRequestDone(EtcdClient::Response* resp, Task* parent_task,
                       EtcdClient::GenericResponse* gen_resp, Task* task) {
  *resp = EtcdClient::Response();
  if (!task->status().ok()) {
    parent_task->Return(task->status());
    return;
  }

  const JsonObject json_node(*gen_resp->json_body, "node");
  if (!json_node.Ok()) {
    parent_task->Return(Status(util::error::FAILED_PRECONDITION,
                               "Invalid JSON: Couldn't find 'node'"));
    return;
  }

  StatusOr<EtcdClient::Node> node(ParseNodeFromJson(json_node));
  if (!node.status().ok()) {
    parent_task->Return(node.status());
    return;
  }

  resp->etcd_index = node.ValueOrDie().modified_index_;
  parent_task->Return();
}


void ForceSetRequestDone(EtcdClient::Response* resp, Task* parent_task,
                         EtcdClient::GenericResponse* gen_resp, Task* task) {
  *resp = EtcdClient::Response();
  if (!task->status().ok()) {
    parent_task->Return(task->status());
    return;
  }

  const JsonObject json_node(*gen_resp->json_body, "node");
  if (!json_node.Ok()) {
    parent_task->Return(Status(util::error::FAILED_PRECONDITION,
                               "Invalid JSON: Couldn't find 'node'"));
    return;
  }

  StatusOr<EtcdClient::Node> node(ParseNodeFromJson(json_node));
  if (!node.status().ok()) {
    parent_task->Return(node.status());
    return;
  }

  resp->etcd_index = node.ValueOrDie().modified_index_;
  parent_task->Return();
}


string UrlEscapeAndJoinParams(const map<string, string>& params) {
  string retval;

  bool first(true);
  for (map<string, string>::const_iterator it = params.begin();
       it != params.end(); ++it) {
    if (first)
      first = false;
    else
      retval += "&";

    unique_ptr<char, void (*)(void*)> first(
        evhttp_uriencode(it->first.c_str(), it->first.size(), 0), &free);
    unique_ptr<char, void (*)(void*)> second(
        evhttp_uriencode(it->second.c_str(), it->second.size(), 0), &free);

    retval += first.get();
    retval += "=";
    retval += second.get();
  }

  return retval;
}


static const EtcdClient::Node kInvalidNode(-1, -1, "", false, "", {}, true);


}  // namespace


struct EtcdClient::RequestState {
  RequestState(UrlFetcher::Verb verb, const string& key,
               const string& key_space, map<string, string> params,
               const HostPortPair& host_port, GenericResponse* gen_resp,
               Task* parent_task)
      : gen_resp_(CHECK_NOTNULL(gen_resp)),
        parent_task_(CHECK_NOTNULL(parent_task)) {
    CHECK(!key.empty());
    CHECK_EQ(key[0], '/');

    req_.verb = verb;
    SetHostPort(host_port);

    if (FLAGS_etcd_consistent) {
      params.insert(make_pair("consistent", "true"));
    } else {
      LOG_EVERY_N(WARNING, 100) << "Sending request without 'consistent=true'";
    }
    if (FLAGS_etcd_quorum) {
      params.insert(make_pair("quorum", "true"));
    } else {
      LOG_EVERY_N(WARNING, 100) << "Sending request without 'quorum=true'";
    }

    req_.url.SetPath(key_space + key);
    switch (req_.verb) {
      case UrlFetcher::Verb::POST:
      case UrlFetcher::Verb::PUT:
        req_.headers.insert(
            make_pair("Content-Type", "application/x-www-form-urlencoded"));
        req_.body = UrlEscapeAndJoinParams(params);
        break;

      default:
        req_.url.SetQuery(UrlEscapeAndJoinParams(params));
    }
    VLOG(2) << "path query: " << req_.url.PathQuery();
  }

  void SetHostPort(const HostPortPair& host_port) {
    CHECK(!host_port.first.empty());
    CHECK_GT(host_port.second, 0);
    req_.url.SetProtocol("http");
    req_.url.SetHost(host_port.first);
    req_.url.SetPort(host_port.second);
  }

  GenericResponse* const gen_resp_;
  Task* const parent_task_;

  UrlFetcher::Request req_;
  UrlFetcher::Response resp_;
};


struct EtcdClient::WatchState {
  WatchState(const string& key, const WatchCallback& cb, Task* task)
      : key_(key),
        cb_(cb),
        task_(CHECK_NOTNULL(task)),
        highest_index_seen_(-1) {
  }

  ~WatchState() {
    VLOG(1) << "EtcdClient::Watch: no longer watching " << key_;
  }

  const string key_;
  const WatchCallback cb_;
  Task* const task_;

  int64_t highest_index_seen_;
  map<string, int64_t> known_keys_;
};


void EtcdClient::WatchInitialGetDone(WatchState* state, GetResponse* resp,
                                     Task* task) {
  unique_ptr<GetResponse> resp_deleter(resp);
  if (state->task_->CancelRequested()) {
    state->task_->Return(Status::CANCELLED);
    return;
  }

  // TODO(pphaneuf): Need better error handling here. Have to review
  // what the possible errors are, for now we'll just retry after a delay.
  if (!task->status().ok()) {
    LOG(WARNING) << "Initial get error: " << task->status() << ", will retry "
                 << "in " << FLAGS_etcd_watch_error_retry_delay_seconds
                 << " second(s)";
    state->task_->executor()->Delay(
        seconds(FLAGS_etcd_watch_error_retry_delay_seconds),
        state->task_->AddChild([this, state](Task*) {
          this->WatchRequestDone(state, nullptr, nullptr);
        }));
    return;
  }

  state->highest_index_seen_ =
      max(state->highest_index_seen_, resp->etcd_index);

  vector<Node> nodes;
  if (resp->node.is_dir_) {
    nodes = move(resp->node.nodes_);
  } else {
    nodes.push_back(resp->node);
  }

  vector<Node> updates;
  map<string, int64_t> new_known_keys;
  VLOG(1) << "WatchGet " << state << " : num updates = " << nodes.size();
  for (const auto& node : nodes) {
    // This simply shouldn't happen, but since I think it shouldn't
    // prevent us from continuing processing, CHECKing on this would
    // just be mean...
    LOG_IF(WARNING, resp->etcd_index < node.modified_index_)
        << "X-Etcd-Index (" << resp->etcd_index
        << ") smaller than node modifiedIndex (" << node.modified_index_
        << ") for key \"" << node.key_ << "\"";

    map<string, int64_t>::iterator it(state->known_keys_.find(node.key_));
    if (it == state->known_keys_.end() || it->second < node.modified_index_) {
      VLOG(1) << "WatchGet " << state << " : updated node " << node.key_
              << " @ " << node.modified_index_;
      // Nodes received in an initial get should *always* exist!
      CHECK(!node.deleted_);
      updates.emplace_back(node);
    }

    new_known_keys[node.key_] = node.modified_index_;
    if (it != state->known_keys_.end()) {
      VLOG(1) << "WatchGet " << state << " : stale update " << node.key_
              << " @ " << node.modified_index_;
      state->known_keys_.erase(it);
    }
  }

  // The keys still in known_keys_ at this point have been deleted.
  for (const auto& key : state->known_keys_) {
    // TODO(pphaneuf): Passing in -1 for the created and modified
    // indices, is that a problem? We do have a "last known" modified
    // index in key.second...
    updates.emplace_back(Node(-1, -1, key.first, false, "", {}, true));
  }

  state->known_keys_.swap(new_known_keys);

  SendWatchUpdates(state, move(updates));
}


void EtcdClient::WatchRequestDone(WatchState* state, GetResponse* get_resp,
                                  Task* child_task) {
  // We clean up this way instead of using util::Task::DeleteWhenDone,
  // because our task is long-lived, and we do not want to accumulate
  // these objects.
  unique_ptr<GetResponse> get_resp_deleter(get_resp);

  if (state->task_->CancelRequested()) {
    state->task_->Return(Status::CANCELLED);
    return;
  }

  // Handle when the request index is too old, we have to restart the
  // watch logic (or start the watch logic the first time).
  if (!child_task ||
      (child_task->status().CanonicalCode() == util::error::ABORTED &&
       get_resp->etcd_index >= 0)) {
    // On the first time here, we don't actually have a gen_resp, we
    // just want to start the watch logic.
    if (get_resp) {
      VLOG(1) << "etcd index: " << get_resp->etcd_index;
      state->highest_index_seen_ =
          max(state->highest_index_seen_, get_resp->etcd_index);
    }

    GetResponse* const resp(new GetResponse);
    Get(state->key_, resp,
        state->task_->AddChild(
            bind(&EtcdClient::WatchInitialGetDone, this, state, resp, _1)));

    return;
  }

  if (!child_task->status().ok()) {
    VLOG(1) << "Watch request errored: " << child_task->status();
    StartWatchRequest(state);
    return;
  }

  vector<Node> updates;
  state->highest_index_seen_ =
      max(state->highest_index_seen_, get_resp->node.modified_index_);
  updates.emplace_back(get_resp->node);

  if (!get_resp->node.deleted_) {
    state->known_keys_[get_resp->node.key_] = get_resp->node.modified_index_;
  } else {
    VLOG(1) << "erased key: " << get_resp->node.key_;
    state->known_keys_.erase(get_resp->node.key_);
  }

  SendWatchUpdates(state, move(updates));
}


// This method should always be called on the executor of
// state->task_.
void EtcdClient::SendWatchUpdates(WatchState* state,
                                  const vector<Node>& updates) {
  if (!updates.empty() || state->highest_index_seen_ == -1) {
    state->cb_(updates);
  }

  // Only start the next request once the callback has return, to make
  // sure they are always delivered in order.
  StartWatchRequest(state);
}


void EtcdClient::StartWatchRequest(WatchState* state) {
  if (state->task_->CancelRequested()) {
    state->task_->Return(Status::CANCELLED);
    return;
  }

  Request req(state->key_);
  req.recursive = true;
  req.wait_index = state->highest_index_seen_ + 1;

  GetResponse* const get_resp(new GetResponse);
  Get(req, get_resp, state->task_->AddChild(bind(&EtcdClient::WatchRequestDone,
                                                 this, state, get_resp, _1)));
}


EtcdClient::Node::Node(int64_t created_index, int64_t modified_index,
                       const string& key, bool is_dir, const string& value,
                       vector<Node>&& nodes, bool deleted)
    : created_index_(created_index),
      modified_index_(modified_index),
      key_(key),
      is_dir_(is_dir),
      value_(value),
      nodes_(move(nodes)),
      expires_(system_clock::time_point::max()),
      deleted_(deleted) {
  CHECK(!deleted_ || value_.empty());
  CHECK(!deleted_ || nodes_.empty());
  CHECK(!is_dir_ || value_.empty());
  CHECK(is_dir_ || nodes_.empty());
}


// static
const EtcdClient::Node& EtcdClient::Node::InvalidNode() {
  return kInvalidNode;
}


string EtcdClient::Node::ToString() const {
  ostringstream oss;
  oss << "[" << key_ << ": '" << value_ << "' c: " << created_index_
      << " m: " << modified_index_;
  if (HasExpiry()) {
    time_t time_c = system_clock::to_time_t(expires_);
    oss << " expires: " << ctime(&time_c);
  }

  oss << " dir: " << is_dir_ << " deleted: " << deleted_ << "]";
  return oss.str();
}


bool EtcdClient::Node::HasExpiry() const {
  return expires_ < system_clock::time_point::max();
}


EtcdClient::EtcdClient(Executor* executor, UrlFetcher* fetcher,
                       const string& host, uint16_t port)
    : EtcdClient(executor, fetcher,
                 list<HostPortPair>{HostPortPair(host, port)}) {
}


EtcdClient::EtcdClient(Executor* executor, UrlFetcher* fetcher,
                       const list<HostPortPair>& etcds)
    : executor_(CHECK_NOTNULL(executor)),
      log_version_task_(new SyncTask(executor_)),
      fetcher_(CHECK_NOTNULL(fetcher)),
      etcds_(etcds),
      logged_version_(false) {
  CHECK(!etcds_.empty()) << "No etcd hosts provided.";
  VLOG(1) << "EtcdClient: " << this;

  for (const auto& e : etcds_) {
    CHECK(!e.first.empty()) << "Empty host specified";
    CHECK_GT(e.second, 0) << "Invalid port specified";
  }
}


EtcdClient::HostPortPair EtcdClient::ChooseNextServer() {
  lock_guard<mutex> lock(lock_);

  etcds_.emplace_back(etcds_.front());
  etcds_.pop_front();

  LOG(INFO) << "Selected new etcd server: " << etcds_.front().first << ":"
            << etcds_.front().second;
  return etcds_.front();
}


EtcdClient::EtcdClient()
    : executor_(nullptr), log_version_task_(nullptr), fetcher_(nullptr) {
}


EtcdClient::~EtcdClient() {
  VLOG(1) << "~EtcdClient: " << this;
  if (log_version_task_) {
    log_version_task_->task()->Return();
    log_version_task_->Wait();
  }
}


void EtcdClient::FetchDone(RequestState* etcd_req, Task* task) {
  VLOG(2) << "EtcdClient::FetchDone: " << task->status();

  if (!task->status().ok()) {
    if (task->status().error_code() == util::error::UNAVAILABLE) {
      // Seems etcd wasn't available; pick a new etcd server and retry
      LOG(WARNING) << "Etcd fetch failed: " << task->status() << ", retrying "
                   << "on next etcd server.";
      etcd_req->SetHostPort(ChooseNextServer());
      fetcher_->Fetch(etcd_req->req_, &etcd_req->resp_,
                      etcd_req->parent_task_->AddChild(
                          bind(&EtcdClient::FetchDone, this, etcd_req, _1)));
      return;
    }
    // Otherwise just let the requestor know.
    etcd_req->parent_task_->Return(task->status());
    return;
  }

  VLOG(2) << "response:\n" << etcd_req->resp_;

  if (etcd_req->resp_.status_code == 307) {
    UrlFetcher::Headers::const_iterator it(
        etcd_req->resp_.headers.find("location"));

    if (it == etcd_req->resp_.headers.end()) {
      etcd_req->parent_task_->Return(
          Status(util::error::INTERNAL,
                 "etcd returned a redirect without a Location header?"));
      return;
    }

    const URL url(it->second);
    if (url.Host().empty() || url.Port() == 0) {
      etcd_req->parent_task_->Return(
          Status(util::error::INTERNAL,
                 "could not parse Location header from etcd: " + it->second));
      return;
    }

    etcd_req->SetHostPort(
        UpdateEndpoint(HostPortPair(url.Host(), url.Port())));

    MaybeLogEtcdVersion();

    fetcher_->Fetch(etcd_req->req_, &etcd_req->resp_,
                    etcd_req->parent_task_->AddChild(
                        bind(&EtcdClient::FetchDone, this, etcd_req, _1)));
    return;
  }

  etcd_req->gen_resp_->json_body =
      make_shared<JsonObject>(etcd_req->resp_.body);
  CHECK_NOTNULL(etcd_req->gen_resp_->json_body.get());
  if (!etcd_req->gen_resp_->json_body->Ok()) {
    LOG(WARNING) << "Got invalid JSON: " << etcd_req->resp_.body;
  }

  etcd_req->gen_resp_->etcd_index = -1;

  UrlFetcher::Headers::const_iterator it(
      etcd_req->resp_.headers.find("X-Etcd-Index"));
  if (it != etcd_req->resp_.headers.end()) {
    etcd_req->gen_resp_->etcd_index = atoll(it->second.c_str());
  }

  etcd_req->parent_task_->Return(
      StatusFromResponse(etcd_req->resp_.status_code,
                         *etcd_req->gen_resp_->json_body));
}


EtcdClient::HostPortPair EtcdClient::GetEndpoint() const {
  lock_guard<mutex> lock(lock_);
  return etcds_.front();
}


EtcdClient::HostPortPair EtcdClient::UpdateEndpoint(
    HostPortPair&& new_endpoint) {
  lock_guard<mutex> lock(lock_);
  logged_version_ = false;
  auto it(find(etcds_.begin(), etcds_.end(), new_endpoint));
  if (it == etcds_.end()) {
    // TODO(alcutter): We don't really have a way of knowing when to remove
    // etcd endpoints at the moment.  We should really be querying the etcd
    // cluster for its members and using that list.
    etcds_.emplace_front(move(new_endpoint));
  } else {
    etcds_.splice(etcds_.begin(), etcds_, it);
  }

  LOG(INFO) << "Selected new etcd server: " << etcds_.front().first << ":"
            << etcds_.front().second;
  return etcds_.front();
}


void EtcdClient::Get(const Request& req, GetResponse* resp, Task* task) {
  map<string, string> params;
  if (req.recursive) {
    params["recursive"] = "true";
  }
  if (req.wait_index > 0) {
    params["wait"] = "true";
    params["waitIndex"] = to_string(req.wait_index);
    // TODO(pphaneuf): This is a hack, as "wait" is not incompatible
    // with "quorum=true". It should be left to the caller, though
    // (and I'm not sure defaulting to "quorum=true" is that good an
    // idea, even).
    params["quorum"] = "false";
  }
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);
  Generic(req.key, kKeysSpace, params, UrlFetcher::Verb::GET, gen_resp,
          task->AddChild(
              bind(&GetRequestDone, req.key, resp, task, gen_resp, _1)));
}


void EtcdClient::Create(const string& key, const string& value, Response* resp,
                        Task* task) {
  map<string, string> params;
  params["value"] = value;
  params["prevExist"] = "false";
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);
  Generic(key, kKeysSpace, params, UrlFetcher::Verb::PUT, gen_resp,
          task->AddChild(bind(&CreateRequestDone, resp, task, gen_resp, _1)));
}


void EtcdClient::CreateWithTTL(const string& key, const string& value,
                               const seconds& ttl, Response* resp,
                               Task* task) {
  map<string, string> params;
  params["value"] = value;
  params["prevExist"] = "false";
  params["ttl"] = to_string(ttl.count());
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);
  Generic(key, kKeysSpace, params, UrlFetcher::Verb::PUT, gen_resp,
          task->AddChild(bind(&CreateRequestDone, resp, task, gen_resp, _1)));
}


void EtcdClient::Update(const string& key, const string& value,
                        const int64_t previous_index, Response* resp,
                        Task* task) {
  map<string, string> params;
  params["value"] = value;
  params["prevIndex"] = to_string(previous_index);
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);
  Generic(key, kKeysSpace, params, UrlFetcher::Verb::PUT, gen_resp,
          task->AddChild(bind(&UpdateRequestDone, resp, task, gen_resp, _1)));
}


void EtcdClient::UpdateWithTTL(const string& key, const string& value,
                               const seconds& ttl,
                               const int64_t previous_index, Response* resp,
                               Task* task) {
  map<string, string> params;
  params["value"] = value;
  params["prevIndex"] = to_string(previous_index);
  params["ttl"] = to_string(ttl.count());
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);
  Generic(key, kKeysSpace, params, UrlFetcher::Verb::PUT, gen_resp,
          task->AddChild(bind(&UpdateRequestDone, resp, task, gen_resp, _1)));
}


void EtcdClient::ForceSet(const string& key, const string& value,
                          Response* resp, Task* task) {
  map<string, string> params;
  params["value"] = value;
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);
  Generic(key, kKeysSpace, params, UrlFetcher::Verb::PUT, gen_resp,
          task->AddChild(
              bind(&ForceSetRequestDone, resp, task, gen_resp, _1)));
}


void EtcdClient::ForceSetWithTTL(const string& key, const string& value,
                                 const seconds& ttl, Response* resp,
                                 Task* task) {
  map<string, string> params;
  params["value"] = value;
  params["ttl"] = to_string(ttl.count());
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);
  Generic(key, kKeysSpace, params, UrlFetcher::Verb::PUT, gen_resp,
          task->AddChild(
              bind(&ForceSetRequestDone, resp, task, gen_resp, _1)));
}


void EtcdClient::Delete(const string& key, const int64_t current_index,
                        Task* task) {
  map<string, string> params;
  params["prevIndex"] = to_string(current_index);
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);

  Generic(key, kKeysSpace, params, UrlFetcher::Verb::DELETE, gen_resp, task);
}


void EtcdClient::ForceDelete(const string& key, Task* task) {
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);

  Generic(key, kKeysSpace, map<string, string>(), UrlFetcher::Verb::DELETE,
          gen_resp, task);
}


void EtcdClient::GetStoreStats(StatsResponse* resp, Task* task) {
  map<string, string> params;
  GenericResponse* const gen_resp(new GenericResponse);
  task->DeleteWhenDone(gen_resp);

  Generic(kStoreStatsKey, kStatsSpace, params, UrlFetcher::Verb::GET, gen_resp,
          task->AddChild(
              bind(&GetStoreStatsRequestDone, resp, task, gen_resp, _1)));
}


void EtcdClient::Watch(const string& key, const WatchCallback& cb,
                       Task* task) {
  VLOG(1) << "EtcdClient::Watch: " << key;

  WatchState* const state(new WatchState(key, cb, task));
  task->DeleteWhenDone(state);

  // This will kick off the watch logic, with an initial get request.
  WatchRequestDone(state, nullptr, nullptr);
}


void EtcdClient::Generic(const string& key, const string& key_space,
                         const map<string, string>& params,
                         UrlFetcher::Verb verb, GenericResponse* resp,
                         Task* task) {
  MaybeLogEtcdVersion();
  RequestState* const etcd_req(new RequestState(verb, key, key_space, params,
                                                GetEndpoint(), resp, task));
  task->DeleteWhenDone(etcd_req);

  fetcher_->Fetch(etcd_req->req_, &etcd_req->resp_,
                  etcd_req->parent_task_->AddChild(
                      bind(&EtcdClient::FetchDone, this, etcd_req, _1)));
}

list<EtcdClient::HostPortPair> SplitHosts(const string& hosts_string) {
  vector<string> hosts(util::split(hosts_string, ','));

  list<EtcdClient::HostPortPair> ret;
  for (const auto& h : hosts) {
    // First check that the entire etcd URL can be parsed by evhttp
    unique_ptr<evhttp_uri, void (*)(evhttp_uri*)> uri(
        evhttp_uri_parse(h.c_str()), &evhttp_uri_free);

    if (!uri) {
      LOG(FATAL) << "Invalid etcd_server url specified: " << h;
    }

    vector<string> hp(util::split(h, ':'));
    CHECK_EQ(static_cast<size_t>(2), hp.size())
        << "Invalid host:port string: '" << h << "'";
    const int port(stoi(hp[1]));
    CHECK_LT(0, port) << "Port is <= 0";
    CHECK_GE(65535, port) << "Port is > 65535";

    ret.emplace_back(EtcdClient::HostPortPair(hp[0], port));
  }
  return ret;
}

void EtcdClient::MaybeLogEtcdVersion() {
  lock_guard<mutex> lock(lock_);
  if (logged_version_) {
    return;
  }
  logged_version_ = true;

  const UrlFetcher::Request req(URL("http://" + etcds_.front().first + ":" +
                                    to_string(etcds_.front().second) +
                                    "/version"));
  UrlFetcher::Response* const resp(new UrlFetcher::Response);
  fetcher_->Fetch(req, resp, log_version_task_->task()->AddChild([this, resp](
                                 Task* child_task) {
    unique_ptr<UrlFetcher::Response> resp_deleter(resp);
    if (!child_task->status().ok()) {
      LOG(WARNING) << "Failed to fetch etcd version: " << child_task->status();
    } else {
      LOG(INFO) << "Etcd version: " << resp->body;
    }
  }));
}


}  // namespace cert_trans
