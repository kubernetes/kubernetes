#ifndef CERT_TRANS_CLIENT_ASYNC_LOG_CLIENT_H_
#define CERT_TRANS_CLIENT_ASYNC_LOG_CLIENT_H_

#include <stdint.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "net/url_fetcher.h"
#include "proto/ct.pb.h"

namespace util {
class Executor;
}  // namespace util

namespace cert_trans {


class Cert;
class CertChain;
class PreCertChain;


class AsyncLogClient {
 public:
  enum Status {
    OK,
    CONNECT_FAILED,
    BAD_RESPONSE,
    INTERNAL_ERROR,
    UNKNOWN_ERROR,
    UPLOAD_FAILED,
    INVALID_INPUT,
  };

  struct Entry {
    ct::MerkleTreeLeaf leaf;
    ct::LogEntry entry;
    std::unique_ptr<ct::SignedCertificateTimestamp> sct;
  };

  typedef std::function<void(Status)> Callback;

  // The "executor" will be used to run callbacks.
  // TODO(pphaneuf): The executor would not be necessary if we
  // converted this API to use util::Task.
  // TODO(pphaneuf): Might also want to take a URL object directly,
  // instead of a string?
  AsyncLogClient(util::Executor* const executor, UrlFetcher* fetcher,
                 const std::string& server_uri);

  void GetSTH(ct::SignedTreeHead* sth, const Callback& done);

  // This does not clear "roots" before appending to it.
  void GetRoots(std::vector<std::unique_ptr<Cert>>* roots,
                const Callback& done);

  // This does not clear "entries" before appending the retrieved
  // entries.
  void GetEntries(int first, int last, std::vector<Entry>* entries,
                  const Callback& done);

  // This is NON-standard, and only works with SuperDuper logs.
  // It's intended for internal use when running in a clustered configuration.
  // This does not clear "entries" before appending the retrieved
  // entries.
  void GetEntriesAndSCTs(int first, int last, std::vector<Entry>* entries,
                         const Callback& done);

  void QueryInclusionProof(const ct::SignedTreeHead& sth,
                           const std::string& merkle_leaf_hash,
                           ct::MerkleAuditProof* proof, const Callback& done);

  // This does not clear "proof" before appending to it.
  void GetSTHConsistency(int64_t first, int64_t second,
                         std::vector<std::string>* proof,
                         const Callback& done);

  // Note: these methods can call "done" inline (before they return),
  // if there is a problem with the (pre-)certificate chain.
  void AddCertChain(const CertChain& cert_chain,
                    ct::SignedCertificateTimestamp* sct, const Callback& done);
  void AddPreCertChain(const PreCertChain& pre_cert_chain,
                       ct::SignedCertificateTimestamp* sct,
                       const Callback& done);

 private:
  URL GetURL(const std::string& subpath) const;

  void InternalGetEntries(int first, int last, std::vector<Entry>* entries,
                          bool request_scts, const Callback& done);

  void InternalAddChain(const CertChain& cert_chain,
                        ct::SignedCertificateTimestamp* sct, bool pre_cert,
                        const Callback& done);

  util::Executor* const executor_;
  UrlFetcher* const fetcher_;
  const URL server_url_;

  DISALLOW_COPY_AND_ASSIGN(AsyncLogClient);
};


}  // namespace cert_trans


#endif  // CERT_TRANS_CLIENT_ASYNC_LOG_CLIENT_H_
