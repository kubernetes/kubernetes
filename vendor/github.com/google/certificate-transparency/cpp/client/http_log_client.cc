/* -*- indent-tabs-mode: nil -*- */
#include "client/http_log_client.h"

#include <event2/buffer.h>
#include <glog/logging.h>
#include <functional>
#include <memory>

#include "log/cert.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/json_wrapper.h"
#include "util/util.h"

namespace libevent = cert_trans::libevent;

using cert_trans::AsyncLogClient;
using cert_trans::Cert;
using cert_trans::CertChain;
using cert_trans::HTTPLogClient;
using cert_trans::PreCertChain;
using cert_trans::ThreadPool;
using ct::MerkleAuditProof;
using ct::SignedCertificateTimestamp;
using ct::SignedTreeHead;
using std::bind;
using std::unique_ptr;
using std::placeholders::_1;
using std::string;
using std::vector;

namespace {


void DoneRequest(AsyncLogClient::Status status, AsyncLogClient::Status* retval,
                 bool* done) {
  *retval = status;
  *done = true;
}


}  // namespace

HTTPLogClient::HTTPLogClient(const string& server)
    : base_(new libevent::Base()),
      pool_(),
      fetcher_(base_.get(), &pool_),
      client_(base_.get(), &fetcher_, server) {
}

AsyncLogClient::Status HTTPLogClient::UploadSubmission(
    const string& submission, bool pre, SignedCertificateTimestamp* sct) {
  AsyncLogClient::Status retval(AsyncLogClient::UNKNOWN_ERROR);
  bool done(false);

  if (pre) {
    PreCertChain pre_cert_chain(submission);
    client_.AddPreCertChain(pre_cert_chain, sct,
                            bind(&DoneRequest, _1, &retval, &done));
  } else {
    CertChain cert_chain(submission);
    client_.AddCertChain(cert_chain, sct,
                         bind(&DoneRequest, _1, &retval, &done));
  }

  while (!done) {
    base_->DispatchOnce();
  }

  return retval;
}


AsyncLogClient::Status HTTPLogClient::GetSTH(SignedTreeHead* sth) {
  AsyncLogClient::Status retval(AsyncLogClient::UNKNOWN_ERROR);
  bool done(false);

  client_.GetSTH(sth, bind(&DoneRequest, _1, &retval, &done));
  while (!done) {
    base_->DispatchOnce();
  }

  return retval;
}

AsyncLogClient::Status HTTPLogClient::GetRoots(
    vector<unique_ptr<Cert>>* roots) {
  AsyncLogClient::Status retval(AsyncLogClient::UNKNOWN_ERROR);
  bool done(false);

  client_.GetRoots(roots, bind(&DoneRequest, _1, &retval, &done));
  while (!done) {
    base_->DispatchOnce();
  }

  return retval;
}

AsyncLogClient::Status HTTPLogClient::QueryAuditProof(
    const string& merkle_leaf_hash, MerkleAuditProof* proof) {
  AsyncLogClient::Status retval(AsyncLogClient::UNKNOWN_ERROR);
  bool done(false);
  SignedTreeHead sth;

  client_.GetSTH(&sth, bind(&DoneRequest, _1, &retval, &done));
  while (!done) {
    base_->DispatchOnce();
  }

  if (retval != AsyncLogClient::OK)
    return retval;

  retval = AsyncLogClient::UNKNOWN_ERROR;
  done = false;
  client_.QueryInclusionProof(sth, merkle_leaf_hash, proof,
                              bind(&DoneRequest, _1, &retval, &done));

  while (!done) {
    base_->DispatchOnce();
  }

  return retval;
}

AsyncLogClient::Status HTTPLogClient::GetEntries(
    int first, int last, vector<AsyncLogClient::Entry>* entries) {
  AsyncLogClient::Status retval(AsyncLogClient::UNKNOWN_ERROR);
  bool done(false);

  client_.GetEntries(first, last, entries,
                     bind(&DoneRequest, _1, &retval, &done));
  while (!done) {
    base_->DispatchOnce();
  }

  return retval;
}

AsyncLogClient::Status HTTPLogClient::GetSTHConsistency(
    int64_t size1, int64_t size2, vector<string>* proof) {
  AsyncLogClient::Status retval(AsyncLogClient::UNKNOWN_ERROR);
  bool done(false);

  client_.GetSTHConsistency(size1, size2, proof,
                            bind(&DoneRequest, _1, &retval, &done));
  while (!done) {
    base_->DispatchOnce();
  }

  return retval;
}
