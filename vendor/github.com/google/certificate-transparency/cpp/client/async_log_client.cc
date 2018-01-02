#include "client/async_log_client.h"

#include <event2/http.h>
#include <glog/logging.h>
#include <algorithm>
#include <iterator>
#include <memory>

#include "log/cert.h"
#include "proto/cert_serializer.h"
#include "proto/serializer.h"
#include "util/json_wrapper.h"

using cert_trans::AsyncLogClient;
using cert_trans::Cert;
using cert_trans::CertChain;
using cert_trans::PreCertChain;
using cert_trans::URL;
using cert_trans::UrlFetcher;
using ct::DigitallySigned;
using ct::MerkleAuditProof;
using ct::SignedCertificateTimestamp;
using ct::SignedTreeHead;
using std::back_inserter;
using std::bind;
using std::move;
using std::placeholders::_1;
using std::string;
using std::to_string;
using std::unique_ptr;
using std::vector;

namespace {


string UriEncode(const string& input) {
  const unique_ptr<char, void (*)(void*)> output(
      evhttp_uriencode(input.data(), input.size(), false), &free);

  return output.get();
}


// Do some common checks, calls the callback with the appropriate
// error if something is wrong.
bool SanityCheck(UrlFetcher::Response* resp,
                 const AsyncLogClient::Callback& done, util::Task* task) {
  // TODO(pphaneuf): We should report errors better. The easiest way
  // would be for this to use util::Task as well, so it could simply
  // pass on the status.
  if (!task->status().ok() || resp->status_code != HTTP_OK) {
    done(AsyncLogClient::UNKNOWN_ERROR);
    return false;
  }

  return true;
}


void DoneGetSTH(UrlFetcher::Response* resp, SignedTreeHead* sth,
                const AsyncLogClient::Callback& done, util::Task* task) {
  unique_ptr<UrlFetcher::Response> resp_deleter(CHECK_NOTNULL(resp));
  unique_ptr<util::Task> task_deleter(CHECK_NOTNULL(task));

  LOG_IF(INFO, !task->status().ok()) << "DoneGetSTH: " << task->status();

  if (!SanityCheck(resp, done, task)) {
    return;
  }

  JsonObject jresponse(resp->body);
  if (!jresponse.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonInt tree_size(jresponse, "tree_size");
  if (!tree_size.Ok() || tree_size.Value() < 0)
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonInt timestamp(jresponse, "timestamp");
  if (!timestamp.Ok() || timestamp.Value() < 0)
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonString root_hash(jresponse, "sha256_root_hash");
  if (!root_hash.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonString jsignature(jresponse, "tree_head_signature");
  if (!jsignature.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);
  DigitallySigned signature;
  if (Deserializer::DeserializeDigitallySigned(jsignature.FromBase64(),
                                               &signature) !=
      DeserializeResult::OK)
    return done(AsyncLogClient::BAD_RESPONSE);

  sth->Clear();
  sth->set_version(ct::V1);
  sth->set_tree_size(tree_size.Value());
  sth->set_timestamp(timestamp.Value());
  sth->set_sha256_root_hash(root_hash.FromBase64());
  sth->mutable_signature()->CopyFrom(signature);

  return done(AsyncLogClient::OK);
}


void DoneGetRoots(UrlFetcher::Response* resp, vector<unique_ptr<Cert>>* roots,
                  const AsyncLogClient::Callback& done, util::Task* task) {
  unique_ptr<UrlFetcher::Response> resp_deleter(CHECK_NOTNULL(resp));
  unique_ptr<util::Task> task_deleter(CHECK_NOTNULL(task));

  if (!SanityCheck(resp, done, task)) {
    return;
  }

  JsonObject jresponse(resp->body);
  if (!jresponse.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonArray jroots(jresponse, "certificates");
  if (!jroots.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  vector<unique_ptr<Cert>> retval;
  for (int i = 0; i < jroots.Length(); ++i) {
    JsonString jcert(jroots, i);
    if (!jcert.Ok())
      return done(AsyncLogClient::BAD_RESPONSE);

    unique_ptr<Cert> cert(new Cert);
    const util::Status status(cert->LoadFromDerString(jcert.FromBase64()));
    if (!status.ok()) {
      return done(AsyncLogClient::BAD_RESPONSE);
    }

    retval.push_back(move(cert));
  }

  roots->swap(retval);

  return done(AsyncLogClient::OK);
}


void DoneGetEntries(UrlFetcher::Response* resp,
                    vector<AsyncLogClient::Entry>* entries,
                    const AsyncLogClient::Callback& done, util::Task* task) {
  unique_ptr<UrlFetcher::Response> resp_deleter(CHECK_NOTNULL(resp));
  unique_ptr<util::Task> task_deleter(CHECK_NOTNULL(task));

  if (!SanityCheck(resp, done, task)) {
    return;
  }

  JsonObject jresponse(resp->body);
  if (!jresponse.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonArray jentries(jresponse, "entries");
  if (!jentries.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  vector<AsyncLogClient::Entry> new_entries;
  new_entries.reserve(jentries.Length());

  for (int n = 0; n < jentries.Length(); ++n) {
    JsonObject entry(jentries, n);
    if (!entry.Ok()) {
      return done(AsyncLogClient::BAD_RESPONSE);
    }

    JsonString leaf_input(entry, "leaf_input");
    if (!leaf_input.Ok()) {
      return done(AsyncLogClient::BAD_RESPONSE);
    }

    AsyncLogClient::Entry log_entry;
    if (Deserializer::DeserializeMerkleTreeLeaf(leaf_input.FromBase64(),
                                                &log_entry.leaf) !=
        DeserializeResult::OK) {
      return done(AsyncLogClient::BAD_RESPONSE);
    }

    JsonString extra_data(entry, "extra_data");
    if (!extra_data.Ok()) {
      return done(AsyncLogClient::BAD_RESPONSE);
    }

    // This is an optional non-standard extension, used only by the log
    // internally when running in clustered mode.
    JsonString sct_data(entry, "sct");
    if (sct_data.Ok()) {
      unique_ptr<SignedCertificateTimestamp> sct(
          new SignedCertificateTimestamp);
      if (Deserializer::DeserializeSCT(sct_data.FromBase64(), sct.get()) !=
          DeserializeResult::OK) {
        return done(AsyncLogClient::BAD_RESPONSE);
      }
      log_entry.sct.reset(sct.release());
    }

    switch (log_entry.leaf.timestamped_entry().entry_type()) {
      case ct::X509_ENTRY:
        DeserializeX509Chain(extra_data.FromBase64(),
                             log_entry.entry.mutable_x509_entry());
        break;
      case ct::PRECERT_ENTRY:
        DeserializePrecertChainEntry(extra_data.FromBase64(),
                                     log_entry.entry.mutable_precert_entry());
        break;
      case ct::X_JSON_ENTRY:
        // nothing to do
        break;
      default:
        LOG(FATAL) << "Don't understand entry type: "
                   << log_entry.leaf.timestamped_entry().entry_type();
    }

    new_entries.emplace_back(move(log_entry));
  }

  entries->reserve(entries->size() + new_entries.size());
  move(new_entries.begin(), new_entries.end(), back_inserter(*entries));

  return done(AsyncLogClient::OK);
}


void DoneQueryInclusionProof(UrlFetcher::Response* resp,
                             const SignedTreeHead& sth,
                             MerkleAuditProof* proof,
                             const AsyncLogClient::Callback& done,
                             util::Task* task) {
  unique_ptr<UrlFetcher::Response> resp_deleter(CHECK_NOTNULL(resp));
  unique_ptr<util::Task> task_deleter(CHECK_NOTNULL(task));

  if (!SanityCheck(resp, done, task)) {
    return;
  }

  JsonObject jresponse(resp->body);
  if (!jresponse.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonInt leaf_index(jresponse, "leaf_index");
  if (!leaf_index.Ok() || leaf_index.Value() < 0)
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonArray audit_path(jresponse, "audit_path");
  if (!audit_path.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  vector<string> path_nodes;
  for (int n = 0; n < audit_path.Length(); ++n) {
    JsonString path_node(audit_path, n);
    CHECK(path_node.Ok());
    path_nodes.push_back(path_node.FromBase64());
  }

  proof->Clear();
  proof->set_version(ct::V1);
  proof->set_tree_size(sth.tree_size());
  proof->set_timestamp(sth.timestamp());
  proof->mutable_tree_head_signature()->CopyFrom(sth.signature());
  proof->set_leaf_index(leaf_index.Value());
  for (vector<string>::const_iterator it = path_nodes.begin();
       it != path_nodes.end(); ++it) {
    proof->add_path_node(*it);
  }

  return done(AsyncLogClient::OK);
}


void DoneGetSTHConsistency(UrlFetcher::Response* resp, vector<string>* proof,
                           const AsyncLogClient::Callback& done,
                           util::Task* task) {
  unique_ptr<UrlFetcher::Response> resp_deleter(CHECK_NOTNULL(resp));
  unique_ptr<util::Task> task_deleter(CHECK_NOTNULL(task));

  if (!SanityCheck(resp, done, task)) {
    return;
  }

  JsonObject jresponse(resp->body);
  if (!jresponse.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonArray jproof(jresponse, "consistency");
  if (!jproof.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  vector<string> entries;
  for (int i = 0; i < jproof.Length(); ++i) {
    JsonString entry(jproof, i);
    if (!entry.Ok())
      return done(AsyncLogClient::BAD_RESPONSE);

    entries.push_back(entry.FromBase64());
  }

  proof->reserve(proof->size() + entries.size());
  move(entries.begin(), entries.end(), back_inserter(*proof));

  return done(AsyncLogClient::OK);
}


void DoneInternalAddChain(UrlFetcher::Response* resp,
                          SignedCertificateTimestamp* sct,
                          const AsyncLogClient::Callback& done,
                          util::Task* task) {
  unique_ptr<UrlFetcher::Response> resp_deleter(CHECK_NOTNULL(resp));
  unique_ptr<util::Task> task_deleter(CHECK_NOTNULL(task));

  if (!SanityCheck(resp, done, task)) {
    return;
  }

  JsonObject jresponse(resp->body);
  if (!jresponse.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  if (!jresponse.IsType(json_type_object))
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonString id(jresponse, "id");
  if (!id.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonInt timestamp(jresponse, "timestamp");
  if (!timestamp.Ok() || timestamp.Value() < 0)
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonString extensions(jresponse, "extensions");
  if (!extensions.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  JsonString jsignature(jresponse, "signature");
  if (!jsignature.Ok())
    return done(AsyncLogClient::BAD_RESPONSE);

  DigitallySigned signature;
  if (Deserializer::DeserializeDigitallySigned(jsignature.FromBase64(),
                                               &signature) !=
      DeserializeResult::OK)
    return done(AsyncLogClient::BAD_RESPONSE);

  sct->Clear();
  sct->set_version(ct::V1);
  sct->mutable_id()->set_key_id(id.FromBase64());
  sct->set_timestamp(timestamp.Value());
  sct->set_extensions(extensions.FromBase64());
  sct->mutable_signature()->CopyFrom(signature);

  return done(AsyncLogClient::OK);
}


URL NormalizeURL(const string& server_url) {
  URL retval(server_url);
  string newpath(retval.Path());

  if (newpath.empty() || newpath.back() != '/')
    newpath.append("/");

  newpath.append("ct/v1/");

  retval.SetPath(newpath);

  return retval;
}


}  // namespace

namespace cert_trans {


AsyncLogClient::AsyncLogClient(util::Executor* const executor,
                               UrlFetcher* fetcher, const string& server_url)
    : executor_(CHECK_NOTNULL(executor)),
      fetcher_(CHECK_NOTNULL(fetcher)),
      server_url_(NormalizeURL(server_url)) {
}


void AsyncLogClient::GetSTH(SignedTreeHead* sth, const Callback& done) {
  UrlFetcher::Response* const resp(new UrlFetcher::Response);
  fetcher_->Fetch(GetURL("get-sth"), resp,
                  new util::Task(bind(DoneGetSTH, resp, sth, done, _1),
                                 executor_));
}


void AsyncLogClient::GetRoots(vector<unique_ptr<Cert>>* roots,
                              const Callback& done) {
  UrlFetcher::Response* const resp(new UrlFetcher::Response);

  fetcher_->Fetch(GetURL("get-roots"), resp,
                  new util::Task(bind(DoneGetRoots, resp, roots, done, _1),
                                 executor_));
}


void AsyncLogClient::GetEntries(int first, int last, vector<Entry>* entries,
                                const Callback& done) {
  return InternalGetEntries(first, last, entries, false /* request_scts */,
                            done);
}


void AsyncLogClient::GetEntriesAndSCTs(int first, int last,
                                       vector<Entry>* entries,
                                       const Callback& done) {
  return InternalGetEntries(first, last, entries, true /* request_scts */,
                            done);
}


void AsyncLogClient::InternalGetEntries(int first, int last,
                                        vector<Entry>* entries,
                                        bool request_scts,
                                        const Callback& done) {
  CHECK_GE(first, 0);
  CHECK_GE(last, 0);

  if (last < first) {
    done(INVALID_INPUT);
    return;
  }

  URL url(GetURL("get-entries"));
  url.SetQuery("start=" + to_string(first) + "&end=" + to_string(last) +
               (request_scts ? "&include_scts=true" : ""));

  UrlFetcher::Response* const resp(new UrlFetcher::Response);
  fetcher_->Fetch(url, resp,
                  new util::Task(bind(DoneGetEntries, resp, entries, done, _1),
                                 executor_));
}


void AsyncLogClient::QueryInclusionProof(const SignedTreeHead& sth,
                                         const std::string& merkle_leaf_hash,
                                         MerkleAuditProof* proof,
                                         const Callback& done) {
  CHECK_GE(sth.tree_size(), 0);

  URL url(GetURL("get-proof-by-hash"));
  url.SetQuery("hash=" + UriEncode(util::ToBase64(merkle_leaf_hash)) +
               "&tree_size=" + to_string(sth.tree_size()));

  UrlFetcher::Response* const resp(new UrlFetcher::Response);
  fetcher_->Fetch(url, resp, new util::Task(bind(DoneQueryInclusionProof, resp,
                                                 sth, proof, done, _1),
                                            executor_));
}


void AsyncLogClient::GetSTHConsistency(int64_t first, int64_t second,
                                       vector<string>* proof,
                                       const Callback& done) {
  CHECK_GE(first, 0);
  CHECK_GE(second, 0);

  URL url(GetURL("get-sth-consistency"));
  url.SetQuery("first=" + to_string(first) + "&second=" + to_string(second));

  UrlFetcher::Response* const resp(new UrlFetcher::Response);
  fetcher_->Fetch(url, resp, new util::Task(bind(DoneGetSTHConsistency, resp,
                                                 proof, done, _1),
                                            executor_));
}


void AsyncLogClient::AddCertChain(const CertChain& cert_chain,
                                  SignedCertificateTimestamp* sct,
                                  const Callback& done) {
  InternalAddChain(cert_chain, sct, false, done);
}


void AsyncLogClient::AddPreCertChain(const PreCertChain& pre_cert_chain,
                                     SignedCertificateTimestamp* sct,
                                     const Callback& done) {
  InternalAddChain(pre_cert_chain, sct, true, done);
}


URL AsyncLogClient::GetURL(const std::string& subpath) const {
  URL retval(server_url_);
  CHECK(!retval.Path().empty());
  CHECK_EQ(retval.Path().back(), '/');
  retval.SetPath(retval.Path() + subpath);
  return retval;
}


void AsyncLogClient::InternalAddChain(const CertChain& cert_chain,
                                      SignedCertificateTimestamp* sct,
                                      bool pre_cert, const Callback& done) {
  if (!cert_chain.IsLoaded())
    return done(INVALID_INPUT);

  JsonArray jchain;
  for (size_t n = 0; n < cert_chain.Length(); ++n) {
    string cert;
    CHECK_EQ(util::Status::OK, cert_chain.CertAt(n)->DerEncoding(&cert));
    jchain.AddBase64(cert);
  }

  JsonObject jsend;
  jsend.Add("chain", jchain);

  UrlFetcher::Request req(GetURL(pre_cert ? "add-pre-chain" : "add-chain"));
  req.verb = UrlFetcher::Verb::POST;
  req.body = jsend.ToString();

  UrlFetcher::Response* const resp(new UrlFetcher::Response);
  fetcher_->Fetch(req, resp, new util::Task(bind(DoneInternalAddChain, resp,
                                                 sct, done, _1),
                                            executor_));
}


}  // namespace cert_trans
