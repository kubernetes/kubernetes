/* -*- indent-tabs-mode: nil -*- */
#include <event2/thread.h>
#include <fcntl.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <openssl/asn1.h>
#include <openssl/bio.h>
#include <openssl/bn.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "client/http_log_client.h"
#include "client/ssl_client.h"
#include "log/cert.h"
#include "log/cert_submission_handler.h"
#include "log/ct_extensions.h"
#include "log/log_signer.h"
#include "log/log_verifier.h"
#include "merkletree/merkle_tree.h"
#include "merkletree/merkle_verifier.h"
#include "merkletree/serial_hasher.h"
#include "monitor/database.h"
#include "monitor/monitor.h"
#include "monitor/sqlite_db.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"
#include "util/init.h"
#include "util/openssl_scoped_types.h"
#include "util/read_key.h"

DEFINE_string(ssl_client_trusted_cert_dir, "",
              "Trusted root certificates for the ssl client");
DEFINE_string(ct_server_public_key, "",
              "PEM-encoded public key file of the CT log server");
DEFINE_string(ssl_server, "", "SSL server to connect to");
DEFINE_int32(ssl_server_port, 0, "SSL server port");
DEFINE_string(ct_server_submission, "",
              "Certificate chain to submit to a CT log server. "
              "The file must consist of concatenated PEM certificates.");
DEFINE_string(ct_server, "", "CT log server to connect to");
DEFINE_string(ct_server_response_out, "",
              "Output file for the Signed Certificate Timestamp received from "
              "the CT log server");
DEFINE_bool(precert, false, "The submission is a CA precertificate chain");
DEFINE_string(sct_token, "",
              "Input file containing the SCT of the certificate");
DEFINE_string(ssl_client_ct_data_in, "",
              "Input file for reading the SSLClientCTData");
DEFINE_string(ssl_client_ct_data_out, "",
              "Output file for recording the server's leaf certificate, "
              "as well as all received and validated SCTs.");
DEFINE_string(certificate_out, "",
              "Output file for the superfluous certificate");
DEFINE_string(tls_extension_data_out, "",
              "Output file for TLS extension data");
DEFINE_string(extensions_config_out, "",
              "Output configuration file to append the sct to. Appends the "
              "sct to the end of the file, so the relevant section should be "
              "last in the configuration file.");
DEFINE_bool(ssl_client_require_sct, true,
            "Fail the SSL handshake if "
            "the server presents no valid SCT token");
DEFINE_bool(ssl_client_expect_handshake_failure, false,
            "Expect the handshake to fail. If this is set to true, then "
            "the program exits with 0 iff there is a handshake failure. "
            "Used for testing.");
DEFINE_string(certificate_chain_in, "",
              "Certificate chain to analyze, "
              "in PEM format");
DEFINE_string(sct_in, "", "SCT to wrap");
DEFINE_int32(get_first, 0, "First entry to retrieve with the 'get' command");
DEFINE_int32(get_last, 0, "Last entry to retrieve with the 'get' command");
DEFINE_string(certificate_base, "",
              "Base name for retrieved certificates - "
              "files will be <base><entry>.<cert>.der");
DEFINE_string(
    monitor_action, "loop",
    "Step the monitor shall do (or loop). "
    "Available actions are:\n"
    "get_sth - put current STH from log into monitor database\n"
    "verify_sth - verify a STH (latest written or for a given timestamp)\n"
    "get_entries - put entries from log into monitor database\n"
    "confirm_tree - build merkletree (latest STH in db OR a given timestamp)\n"
    "init - initiate monitor (i.e. database) prior to its first run\n"
    "loop - start the monitor in a loop (default)");
DEFINE_string(sqlite_db, "", "Database for certificate and tree storage");
DEFINE_uint64(timestamp, 0,
              "The timestamp to be used in the monitor actions "
              "verify_sth and confirm_tree.");
DEFINE_string(sth1, "", "File containing first STH");
DEFINE_string(sth2, "", "File containing second STH");
DEFINE_uint64(monitor_sleep_time_secs, 60,
              "Amount of time the monitor shall "
              "sleep between probing for a new STH.");


static const char kUsage[] =
    " <command> ...\n"
    "Known commands:\n"
    "connect - connect to an SSL server\n"
    "upload - upload a submission to a CT log server\n"
    "certificate - make a superfluous proof certificate\n"
    "extension_data - convert an audit proof to TLS extension format\n"
    "configure_proof - write the proof in an X509v3 configuration file\n"
    "diagnose_chain - print info about the SCTs the cert chain carries\n"
    "wrap - take an SCT and certificate chain and wrap them as if they were\n"
    "       retrieved via 'connect'\n"
    "wrap_embedded - take a certificate chain with an embedded SCT and wrap\n"
    "                them as if they were retrieved via 'connect'\n"
    "get_roots - get roots from the log\n"
    "get_entries - get entries from the log\n"
    "sth - get the current STH from the log\n"
    "consistency - get and check consistency of two STHs\n"
    "monitor - use the monitor (see monitor_action flag)\n"
    "Use --help to display command-line flag options\n";

using cert_trans::AsyncLogClient;
using cert_trans::Cert;
using cert_trans::CertChain;
using cert_trans::CertSubmissionHandler;
using cert_trans::HTTPLogClient;
using cert_trans::PreCertChain;
using cert_trans::ReadPublicKey;
using cert_trans::SSLClient;
using cert_trans::ScopedASN1_OCTET_STRING;
using cert_trans::ScopedBIGNUM;
using cert_trans::ScopedBIO;
using cert_trans::ScopedEVP_PKEY;
using cert_trans::ScopedRSA;
using cert_trans::ScopedX509;
using cert_trans::ScopedX509_NAME;
using cert_trans::TbsCertificate;
using ct::LogEntry;
using ct::MerkleAuditProof;
using ct::SSLClientCTData;
using ct::SignedCertificateTimestamp;
using ct::SignedCertificateTimestampList;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;
using util::StatusOr;

// SCTs presented to clients have to be encoded as a list.
// Helper method for encoding a single SCT.
static string SCTToList(const string& serialized_sct) {
  SignedCertificateTimestampList sct_list;
  sct_list.add_sct_list(serialized_sct);
  string result;
  CHECK_EQ(SerializeResult::OK,
           Serializer::SerializeSCTList(sct_list, &result));
  return result;
}

static LogVerifier* GetLogVerifierFromFlags() {
  CHECK(!FLAGS_ct_server_public_key.empty());

  StatusOr<EVP_PKEY*> pkey(ReadPublicKey(FLAGS_ct_server_public_key));
  CHECK(pkey.ok()) << "could not read CT server public key file: "
                   << pkey.status();

  return new LogVerifier(new LogSigVerifier(pkey.ValueOrDie()),
                         new MerkleVerifier(new Sha256Hasher()));
}

// Adds the data to the cert as an extension, formatted as a single
// ASN.1 octet string.
static void AddOctetExtension(X509* cert, int nid, const unsigned char* data,
                              int data_len, int critical) {
  // The extension as a single octet string.
  ScopedASN1_OCTET_STRING inner(ASN1_OCTET_STRING_new());
  CHECK_NOTNULL(inner.get());
  CHECK_EQ(1, ASN1_OCTET_STRING_set(inner.get(), data, data_len));
  int buf_len = i2d_ASN1_OCTET_STRING(inner.get(), NULL);
  CHECK_GT(buf_len, 0);

  unsigned char buf[buf_len];
  unsigned char* p = buf;

  CHECK_EQ(buf_len, i2d_ASN1_OCTET_STRING(inner.get(), &p));

  // The outer, opaque octet string.
  ScopedASN1_OCTET_STRING asn1_data(ASN1_OCTET_STRING_new());
  CHECK_NOTNULL(asn1_data.get());
  CHECK_EQ(1, ASN1_OCTET_STRING_set(asn1_data.get(), buf, buf_len));

  X509_EXTENSION* ext =
      X509_EXTENSION_create_by_NID(NULL, nid, critical, asn1_data.get());
  CHECK_EQ(1, X509_add_ext(cert, ext, -1));
}

// Reconstructs a LogEntry from the given precert chain.
// Used for verifying a Precert SCT.
// Returns true iff the LogEntry was correctly populated.
static bool PrecertChainToEntry(const cert_trans::PreCertChain& chain,
                                LogEntry* entry) {
  if (!chain.IsLoaded()) {
    LOG(ERROR) << "Chain not loaded.";
    return false;
  }

  const StatusOr<bool> has_poison =
      chain.LeafCert()->HasExtension(cert_trans::NID_ctPoison);
  if (!has_poison.ok()) {
    LOG(ERROR) << "Failed to test for poison extension.";
    return false;
  }

  if (!has_poison.ValueOrDie()) {
    LOG(ERROR) << "Leaf cert doesn't seem to be a Precertificate (no Poison).";
    return false;
  }

  if (chain.Length() < 2) {
    LOG(ERROR) << "Need issuer.";
    return false;
  }

  entry->set_type(ct::PRECERT_ENTRY);
  string key_hash;
  if (!chain.CertAt(1)->SPKISha256Digest(&key_hash).ok()) {
    LOG(ERROR) << "Failed to get SPKISha256.";
    return false;
  }

  entry->mutable_precert_entry()->mutable_pre_cert()->set_issuer_key_hash(
      key_hash);

  TbsCertificate tbs(*chain.LeafCert());
  if (!tbs.IsLoaded()) {
    LOG(ERROR) << "Failed to get TbsCertificate.";
    return false;
  }
  // DeleteExtension can return NOT_FOUND but we checked the extension exists
  // above so this is not expected.
  if (!tbs.DeleteExtension(cert_trans::NID_ctPoison).ok()) {
    LOG(ERROR) << "Failed to delete poison extension.";
    return false;
  }

  string tbs_der;
  if (!tbs.DerEncoding(&tbs_der).ok()) {
    LOG(ERROR) << "Couldn't serialize TbsCertificate to DER.";
    return false;
  }

  entry->mutable_precert_entry()->mutable_pre_cert()->set_tbs_certificate(
      tbs_der);
  return true;
}

static bool VerifySCTAndPopulateSSLClientCTData(
    const SignedCertificateTimestamp& sct, SSLClientCTData* ct_data) {
  SSLClientCTData::SCTInfo* sct_info = ct_data->add_attached_sct_info();
  sct_info->mutable_sct()->CopyFrom(sct);
  const unique_ptr<LogVerifier> verifier(GetLogVerifierFromFlags());
  string merkle_leaf;
  LogVerifier::LogVerifyResult result =
      verifier->VerifySignedCertificateTimestamp(
          ct_data->reconstructed_entry(), sct, &merkle_leaf);
  if (result != LogVerifier::VERIFY_OK) {
    LOG(ERROR) << "Verifier returned " << result;
    return false;
  }

  sct_info->set_merkle_leaf_hash(merkle_leaf);

  return true;
}

// Checks an SCT issued for an X.509 Certificate.
static bool CheckSCT(const SignedCertificateTimestamp& sct,
                     const CertChain& chain, SSLClientCTData* ct_data) {
  LogEntry entry;
  if (!CertSubmissionHandler::X509ChainToEntry(chain, &entry)) {
    LOG(ERROR) << "Failed to reconstruct log entry input from chain";
    return false;
  }
  ct_data->mutable_reconstructed_entry()->CopyFrom(entry);
  return VerifySCTAndPopulateSSLClientCTData(sct, ct_data);
}

// Checks an SCT issued for a Precert.
static bool CheckSCT(const SignedCertificateTimestamp& sct,
                     const PreCertChain& chain, SSLClientCTData* ct_data) {
  LogEntry entry;
  if (!PrecertChainToEntry(chain, &entry)) {
    LOG(ERROR) << "Failed to reconstruct log entry input from precert chain";
    return false;
  }
  ct_data->mutable_reconstructed_entry()->CopyFrom(entry);
  return VerifySCTAndPopulateSSLClientCTData(sct, ct_data);
}

void WriteFile(const std::string& file, const std::string& contents,
               const char* name) {
  if (file.empty()) {
    LOG(WARNING) << "No response file specified; " << name
                 << " will not be saved.";
    return;
  }
  std::ofstream out(file.c_str(), std::ios::out | std::ios::binary);
  PCHECK(out.good()) << "Could not open file " << file << " for writing";
  out.write(contents.data(), contents.size());
  out.close();
  LOG(INFO) << name << " saved in " << file;
}

// Returns true if the server responds with a token; false if
// it responds with an error.
// 0 - ok
// 1 - server says no
// 2 - server unavailable
static int Upload() {
  // Contents should be concatenated PEM entries.
  string contents;
  string submission_file = FLAGS_ct_server_submission;
  PCHECK(util::ReadBinaryFile(submission_file, &contents))
      << "Could not read CT log server submission from " << submission_file;

  LOG(INFO) << "Uploading certificate submission from " << submission_file;
  LOG(INFO) << submission_file << " is " << contents.length() << " bytes.";

  SignedCertificateTimestamp sct;
  HTTPLogClient client(FLAGS_ct_server);
  AsyncLogClient::Status ret =
      client.UploadSubmission(contents, FLAGS_precert, &sct);

  if (ret == AsyncLogClient::CONNECT_FAILED) {
    LOG(ERROR) << "Unable to connect";
    return 2;
  }

  if (ret != AsyncLogClient::OK) {
    LOG(ERROR) << "Submission failed, error = " << ret;
    return 1;
  }

  // Verify the SCT if we can:
  if (FLAGS_precert) {
    SSLClientCTData ct_data;
    PreCertChain chain(contents);
    // Need the issuing cert, otherwise we can't calculate its hash...
    if (chain.Length() > 1) {
      CHECK(CheckSCT(sct, chain, &ct_data));
    } else {
      LOG(WARNING) << "Unable to verify Precert SCT without issuing "
                   << "certificate in chain.";
    }
  } else {
    // SCT for a vanilla X.509 Cert.
    SSLClientCTData ct_data;
    CertChain chain(contents);
    // FIXME: this'll fail if we're uploading a cert which already has an
    // embedded SCT in it, and the issuing cert is not included in the chain
    // since we'll need to create the precert entry under the covers.
    CHECK(CheckSCT(sct, chain, &ct_data));
  }

  // TODO(ekasper): Process the |contents| bundle so that we can verify
  // the token.

  string proof;
  if (Serializer::SerializeSCT(sct, &proof) != SerializeResult::OK) {
    LOG(ERROR) << "Failed to serialize the server token";
    return 1;
  }
  WriteFile(FLAGS_ct_server_response_out, proof, "SCT token");
  return 0;
}

// FIXME: fix all the memory leaks in this code.
static void MakeCert() {
  string sct;
  PCHECK(util::ReadBinaryFile(FLAGS_sct_token, &sct))
      << "Could not read SCT data from " << FLAGS_sct_token;

  string cert_file = FLAGS_certificate_out;

  int cert_fd = open(cert_file.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0666);
  PCHECK(cert_fd > 0) << "Could not open certificate file " << cert_file
                      << " for writing.";

  ScopedBIO out(BIO_new_fd(cert_fd, BIO_CLOSE));

  ScopedX509 x(X509_new());

  // X509v3 (== 2)
  X509_set_version(x.get(), 2);

  // Random 128 bit serial number
  ScopedBIGNUM serial(BN_new());
  BN_rand(serial.get(), 128, 0, 0);
  BN_to_ASN1_INTEGER(serial.get(), X509_get_serialNumber(x.get()));

  // Set signature algorithm
  // FIXME: is there an opaque way to get the algorithm structure?
  x->cert_info->signature->algorithm = OBJ_nid2obj(NID_sha1WithRSAEncryption);
  x->cert_info->signature->parameter = NULL;

  // Set the start date to now
  X509_gmtime_adj(X509_get_notBefore(x.get()), 0);
  // End date to now + 1 second
  X509_gmtime_adj(X509_get_notAfter(x.get()), 1);

  // Create the issuer name
  ScopedX509_NAME issuer(X509_NAME_new());
  X509_NAME_add_entry_by_NID(
      issuer.get(), NID_commonName, V_ASN1_PRINTABLESTRING,
      const_cast<unsigned char*>(
          reinterpret_cast<const unsigned char*>("Test")),
      4, 0, -1);
  X509_set_issuer_name(x.get(), issuer.release());

  // Create the subject name
  ScopedX509_NAME subject(X509_NAME_new());
  X509_NAME_add_entry_by_NID(
      subject.get(), NID_commonName, V_ASN1_PRINTABLESTRING,
      const_cast<unsigned char*>(
          reinterpret_cast<const unsigned char*>("tseT")),
      4, 0, -1);
  X509_set_subject_name(x.get(), subject.release());

  // Public key
  ScopedRSA rsa(RSA_new());
  static const unsigned char bits[1] = {3};
  rsa->n = BN_bin2bn(bits, 1, NULL);
  rsa->e = BN_bin2bn(bits, 1, NULL);
  ScopedEVP_PKEY evp_pkey(EVP_PKEY_new());
  EVP_PKEY_assign_RSA(evp_pkey.get(), rsa.release());
  X509_PUBKEY_set(&X509_get_X509_PUBKEY(x), evp_pkey.release());

  // And finally, the proof in an extension
  const string serialized_sct_list(SCTToList(sct));
  AddOctetExtension(x.get(), cert_trans::NID_ctSignedCertificateTimestampList,
                    reinterpret_cast<const unsigned char*>(
                        serialized_sct_list.data()),
                    serialized_sct_list.size(), 1);

  CHECK_GT(i2d_X509_bio(out.get(), x.get()), 0);
}

// A sample tool for CAs showing how to add the CT proof as an extension.
// We write the CT proof to the certificate config, so that we can
// sign using the standard openssl signing flow.
// Input:
// (1) an X509v3 configuration file
// (2) A binary proof file.
// Output:
// Append the following line to the end of the file.
// (This means the relevant section should be last in the configuration.)
// 1.2.3.1=DER:[raw encoding of proof]
static void WriteProofToConfig() {
  CHECK(!FLAGS_sct_token.empty()) << google::ProgramUsage();
  CHECK(!FLAGS_extensions_config_out.empty()) << google::ProgramUsage();

  string sct;

  PCHECK(util::ReadBinaryFile(FLAGS_sct_token, &sct))
      << "Could not read SCT data from " << FLAGS_sct_token;

  string serialized_sct_list = SCTToList(sct);

  string conf_file = FLAGS_extensions_config_out;

  std::ofstream conf_out(conf_file.c_str(), std::ios::app);
  PCHECK(conf_out.good()) << "Could not open extensions configuration file "
                          << conf_file << " for writing.";

  conf_out << string(cert_trans::kEmbeddedSCTListOID)
           << "=ASN1:FORMAT:HEX,OCTETSTRING:";

  conf_out << util::HexString(serialized_sct_list) << std::endl;
  conf_out.close();
}

static const char kPEMLabel[] = "SERVERINFO FOR SIGNED CERTIFICATE TIMESTAMP";

// Wrap the proof in the format expected by the TLS extension,
// so that we can feed it to OpenSSL.
static void ProofToExtensionData() {
  CHECK(!FLAGS_sct_token.empty()) << google::ProgramUsage();
  CHECK(!FLAGS_tls_extension_data_out.empty()) << google::ProgramUsage();

  string serialized_sct;
  PCHECK(util::ReadBinaryFile(FLAGS_sct_token, &serialized_sct))
      << "Could not read SCT data from " << FLAGS_sct_token;
  std::ifstream proof_in(FLAGS_sct_token.c_str(),
                         std::ios::in | std::ios::binary);
  PCHECK(proof_in.good()) << "Could not read SCT data from "
                          << FLAGS_sct_token;

  // Count proof length.
  proof_in.seekg(0, std::ios::end);
  int proof_length = proof_in.tellg();
  // Rewind.
  proof_in.seekg(0, std::ios::beg);

  // Read the proof
  char* buf = new char[proof_length];
  proof_in.read(buf, proof_length);
  CHECK_EQ(proof_in.gcount(), proof_length);

  SignedCertificateTimestampList sctlist;
  sctlist.add_sct_list(buf, proof_length);
  delete[] buf;

  string sctliststr;
  CHECK_EQ(Serializer::SerializeSCTList(sctlist, &sctliststr),
           SerializeResult::OK);

  std::ostringstream extension_data_out;

  // Write the extension type (18), MSB first.
  extension_data_out << '\0' << '\x12';

  // Write the length, MSB first.
  extension_data_out << static_cast<unsigned char>(sctliststr.length() >> 8)
                     << static_cast<unsigned char>(sctliststr.length());

  // Now write the proof.
  extension_data_out.write(sctliststr.data(), sctliststr.length());
  CHECK(!extension_data_out.bad());

  proof_in.close();

  FILE* out = fopen(FLAGS_tls_extension_data_out.c_str(), "w");
  PCHECK(out != NULL) << "Could not open extension data file "
                      << FLAGS_tls_extension_data_out
                      << " for writing:" << strerror(errno);

// Work around broken PEM_write() declaration in older OpenSSL versions.
#if OPENSSL_VERSION_NUMBER < 0x10002000L
  PEM_write(out, const_cast<char*>(kPEMLabel), const_cast<char*>(""),
            const_cast<unsigned char*>(reinterpret_cast<const unsigned char*>(
                extension_data_out.str().data())),
            extension_data_out.str().length());
#else
  PEM_write(out, kPEMLabel, "", reinterpret_cast<const unsigned char*>(
                                    extension_data_out.str().data()),
            extension_data_out.str().length());
#endif

  fclose(out);
}

static void WriteSSLClientCTData(const SSLClientCTData& ct_data,
                                 const string& ct_data_out_file) {
  std::ofstream checkpoint_out(ct_data_out_file.c_str(),
                               std::ios::out | std::ios::binary);
  PCHECK(checkpoint_out.good()) << "Could not open checkpoint file "
                                << ct_data_out_file << " for writing";
  string serialized_data;
  CHECK(ct_data.SerializeToString(&serialized_data));
  checkpoint_out << serialized_data;
  checkpoint_out.close();
}

// Return values upon completion
//  0: handshake ok
//  1: handshake error
//  2: connection error
static SSLClient::HandshakeResult Connect() {
  LogVerifier* verifier = GetLogVerifierFromFlags();

  SSLClient client(FLAGS_ssl_server, FLAGS_ssl_server_port,
                   FLAGS_ssl_client_trusted_cert_dir, verifier);

  SSLClient::HandshakeResult result;

  if (FLAGS_ssl_client_require_sct)
    result = client.SSLConnectStrict();
  else
    result = client.SSLConnect();

  if (result == SSLClient::OK) {
    SSLClientCTData ct_data;
    client.GetSSLClientCTData(&ct_data);
    if (ct_data.attached_sct_info_size() > 0) {
      LOG(INFO) << "Received " << ct_data.attached_sct_info_size() << " SCTs";
      VLOG(1) << "Received SCTs:";
      for (int i = 0; i < ct_data.attached_sct_info_size(); ++i)
        VLOG(1) << ct_data.attached_sct_info(i).DebugString();
      if (!FLAGS_ssl_client_ct_data_out.empty())
        WriteSSLClientCTData(ct_data, FLAGS_ssl_client_ct_data_out);
    }
  }
  return result;
}

enum AuditResult {
  // At least one SCT has a valid proof.
  // (Should be unusual to have more than one SCT from the same log,
  // but we audit them all and try to see if any are valid).
  PROOF_OK = 0,
  // No SCTs have valid proofs.
  PROOF_NOT_FOUND = 1,
  CT_SERVER_UNAVAILABLE = 2,
};

static AuditResult Audit() {
  string serialized_data;
  PCHECK(util::ReadBinaryFile(FLAGS_ssl_client_ct_data_in, &serialized_data))
      << "Could not read CT data from " << FLAGS_ssl_client_ct_data_in;
  SSLClientCTData ct_data;
  CHECK(ct_data.ParseFromString(serialized_data))
      << "Failed to parse the stored certificate CT data";
  CHECK(ct_data.has_reconstructed_entry());
  CHECK_GT(ct_data.attached_sct_info_size(), 0);

  LogVerifier* verifier = GetLogVerifierFromFlags();
  string key_id = verifier->KeyID();

  AuditResult audit_result = PROOF_NOT_FOUND;

  for (int i = 0; i < ct_data.attached_sct_info_size(); ++i) {
    LOG(INFO) << "Signed Certificate Timestamp number " << i + 1 << ":\n"
              << ct_data.attached_sct_info(i).sct().DebugString();

    string sct_id = ct_data.attached_sct_info(i).sct().id().key_id();
    if (sct_id != key_id) {
      LOG(WARNING) << "Audit skipped: log server Key ID " << sct_id
                   << " does not match verifier's ID";
      continue;
    }

    MerkleAuditProof proof;
    HTTPLogClient client(FLAGS_ct_server);

    LOG(INFO) << "info = " << ct_data.attached_sct_info(i).DebugString();
    AsyncLogClient::Status ret =
        client.QueryAuditProof(ct_data.attached_sct_info(i).merkle_leaf_hash(),
                               &proof);

    // HTTP protocol does not supply this.
    proof.mutable_id()->set_key_id(sct_id);

    if (ret == AsyncLogClient::CONNECT_FAILED) {
      LOG(ERROR) << "Unable to connect";
      delete verifier;
      return CT_SERVER_UNAVAILABLE;
    }
    if (ret != AsyncLogClient::OK) {
      LOG(ERROR) << "QueryAuditProof failed, error " << ret;
      continue;
    }

    LOG(INFO) << "Received proof:\n" << proof.DebugString();
    LogVerifier::LogVerifyResult res =
        verifier->VerifyMerkleAuditProof(ct_data.reconstructed_entry(),
                                         ct_data.attached_sct_info(i).sct(),
                                         proof);
    if (res != LogVerifier::VERIFY_OK) {
      LOG(ERROR) << "Verify error: " << LogVerifier::VerifyResultString(res);
      LOG(ERROR) << "Retrieved Merkle proof is invalid.";
      continue;
    }
    LOG(INFO) << "Proof verified.";
    audit_result = PROOF_OK;
  }
  delete verifier;
  return audit_result;
}

static int CheckConsistency() {
  HTTPLogClient client(FLAGS_ct_server);
  LogVerifier* verifier = GetLogVerifierFromFlags();

  string sth1_str;
  PCHECK(util::ReadBinaryFile(FLAGS_sth1, &sth1_str)) << "Can't read STH file "
                                                      << FLAGS_sth1;
  ct::SignedTreeHead sth1;
  CHECK(sth1.ParseFromString(sth1_str));
  string sth2_str;
  PCHECK(util::ReadBinaryFile(FLAGS_sth2, &sth2_str)) << "Can't read STH file "
                                                      << FLAGS_sth2;
  ct::SignedTreeHead sth2;
  CHECK(sth2.ParseFromString(sth2_str));

  std::vector<string> proof;
  CHECK_EQ(AsyncLogClient::OK,
           client.GetSTHConsistency(sth1.tree_size(), sth2.tree_size(),
                                    &proof));

  if (!verifier->VerifyConsistency(sth1, sth2, proof)) {
    LOG(ERROR) << "Consistency proof does not verify";
    delete verifier;
    return 1;
  }

  LOG(INFO) << "Consistency proof verifies";

  delete verifier;
  return 0;
}

static void DiagnoseCertChain() {
  string cert_file = FLAGS_certificate_chain_in;
  CHECK(!cert_file.empty()) << "Please give a certificate chain with "
                            << "--certificate_chain_in";
  string pem_chain;
  PCHECK(util::ReadBinaryFile(cert_file, &pem_chain))
      << "Could not read certificate chain from " << cert_file;
  CertChain chain(pem_chain);
  CHECK(chain.IsLoaded()) << cert_file
                          << " is not a valid PEM-encoded certificate chain";


  const StatusOr<bool> has_timestamp_list = chain.LeafCert()->HasExtension(
      cert_trans::NID_ctEmbeddedSignedCertificateTimestampList);
  if (!has_timestamp_list.ok() || !has_timestamp_list.ValueOrDie()) {
    LOG(ERROR) << "Certificate has no embedded SCTs";
    return;
  }

  LOG(INFO) << "Embedded proof extension found in certificate";

  unique_ptr<LogVerifier> verifier;
  LogEntry entry;
  if (FLAGS_ct_server_public_key.empty()) {
    LOG(WARNING) << "No log server public key given, skipping verification";
  } else {
    verifier.reset(GetLogVerifierFromFlags());
    CertSubmissionHandler::X509ChainToEntry(chain, &entry);
  }

  string serialized_scts;
  util::Status status = chain.LeafCert()->OctetStringExtensionData(
      cert_trans::NID_ctEmbeddedSignedCertificateTimestampList,
      &serialized_scts);
  if (!status.ok()) {
    LOG(ERROR) << "SCT extension data is missing / invalid.";
    return;
  }

  LOG(INFO) << "Embedded SCT extension length is " << serialized_scts.length()
            << " bytes";

  SignedCertificateTimestampList sct_list;
  if (Deserializer::DeserializeSCTList(serialized_scts, &sct_list) !=
      DeserializeResult::OK) {
    LOG(ERROR) << "Failed to parse SCT list from certificate";
    return;
  }

  LOG(INFO) << "Certificate has " << sct_list.sct_list_size() << " SCTs";
  for (int i = 0; i < sct_list.sct_list_size(); ++i) {
    SignedCertificateTimestamp sct;
    if (Deserializer::DeserializeSCT(sct_list.sct_list(i), &sct) !=
        DeserializeResult::OK) {
      LOG(ERROR) << "Failed to parse SCT number " << i + 1;
      continue;
    }
    LOG(INFO) << "SCT number " << i + 1 << ":\n" << sct.DebugString();
    if (verifier) {
      if (sct.id().key_id() != verifier->KeyID()) {
        LOG(WARNING) << "SCT key ID does not match verifier's ID, skipping";
        continue;
      } else {
        LogVerifier::LogVerifyResult res =
            verifier->VerifySignedCertificateTimestamp(entry, sct);
        if (res == LogVerifier::VERIFY_OK)
          LOG(INFO) << "SCT verified";
        else
          LOG(ERROR) << "SCT verification failed: "
                     << LogVerifier::VerifyResultString(res);
      }
    }
  }
}

// Wrap an SCT in an SSLClientCTData as if it came from an SSL server.
void Wrap() {
  string serialized_data;
  PCHECK(util::ReadBinaryFile(FLAGS_sct_in, &serialized_data))
      << "Could not read SCT data from " << FLAGS_sct_in;
  SignedCertificateTimestamp sct;
  CHECK_EQ(Deserializer::DeserializeSCT(serialized_data, &sct),
           DeserializeResult::OK);

  // FIXME(benl): This code is shared with DiagnoseCertChain().
  string cert_file = FLAGS_certificate_chain_in;
  CHECK(!cert_file.empty()) << "Please give a certificate chain with "
                            << "--certificate_chain_in";
  string pem_chain;
  PCHECK(util::ReadBinaryFile(cert_file, &pem_chain))
      << "Could not read certificate chain from " << cert_file;
  CertChain chain(pem_chain);
  CHECK(chain.IsLoaded()) << cert_file
                          << " is not a valid PEM-encoded certificate chain";

  SSLClientCTData ct_data;
  CHECK(CheckSCT(sct, chain, &ct_data));

  WriteSSLClientCTData(ct_data, FLAGS_ssl_client_ct_data_out);
}

// Wrap an embedded SCT in an SSLClientCTData as if it came from an SSL server.
void WrapEmbedded() {
  // FIXME(benl): This code is shared with DiagnoseCertChain().
  string cert_file = FLAGS_certificate_chain_in;
  CHECK(!cert_file.empty()) << "Please give a certificate chain with "
                            << "--certificate_chain_in";
  string pem_chain;
  PCHECK(util::ReadBinaryFile(cert_file, &pem_chain))
      << "Could not read certificate chain from " << cert_file;
  CertChain chain(pem_chain);
  CHECK(chain.IsLoaded()) << cert_file
                          << " is not a valid PEM-encoded certificate chain";
  CHECK(chain.LeafCert()
            ->HasExtension(
                cert_trans::NID_ctEmbeddedSignedCertificateTimestampList)
            .ValueOrDie());

  string serialized_scts;
  CHECK_EQ(::util::Status::OK,
           chain.LeafCert()->OctetStringExtensionData(
               cert_trans::NID_ctEmbeddedSignedCertificateTimestampList,
               &serialized_scts));
  SignedCertificateTimestampList sct_list;
  CHECK_EQ(DeserializeResult::OK,
           Deserializer::DeserializeSCTList(serialized_scts, &sct_list));

  // FIXME(benl): handle multiple SCTs!
  CHECK_EQ(1, sct_list.sct_list().size());

  SignedCertificateTimestamp sct;
  CHECK_EQ(Deserializer::DeserializeSCT(sct_list.sct_list(0), &sct),
           DeserializeResult::OK);

  SSLClientCTData ct_data;
  CHECK(CheckSCT(sct, chain, &ct_data));

  WriteSSLClientCTData(ct_data, FLAGS_ssl_client_ct_data_out);
}

static void WriteCertificate(const std::string& cert, int entry,
                             int cert_number, const char* type) {
  std::ostringstream outname;
  outname << FLAGS_certificate_base << entry << '.' << cert_number << '.'
          << type << ".der";
  std::ofstream out(outname.str().c_str(), std::ios::binary | std::ios::trunc);
  CHECK(out.good());
  out << cert;
}

void GetEntries() {
  CHECK_NE(FLAGS_ct_server, "");
  HTTPLogClient client(FLAGS_ct_server);
  std::vector<AsyncLogClient::Entry> entries;
  AsyncLogClient::Status error =
      client.GetEntries(FLAGS_get_first, FLAGS_get_last, &entries);
  CHECK_EQ(error, AsyncLogClient::OK);

  CHECK(!FLAGS_certificate_base.empty());

  int e = FLAGS_get_first;
  for (std::vector<AsyncLogClient::Entry>::const_iterator
           entry = entries.begin();
       entry != entries.end(); ++entry, ++e) {
    if (entry->leaf.timestamped_entry().entry_type() == ct::X509_ENTRY) {
      WriteCertificate(entry->leaf.timestamped_entry().signed_entry().x509(),
                       e, 0, "x509");
      const ct::X509ChainEntry& x509chain = entry->entry.x509_entry();
      for (int n = 0; n < x509chain.certificate_chain_size(); ++n)
        WriteCertificate(x509chain.certificate_chain(n), e, n + 1, "x509");
    } else {
      CHECK_EQ(entry->leaf.timestamped_entry().entry_type(),
               ct::PRECERT_ENTRY);
      WriteCertificate(entry->leaf.timestamped_entry()
                           .signed_entry()
                           .precert()
                           .tbs_certificate(),
                       e, 0, "pre");
      const ct::PrecertChainEntry& precertchain = entry->entry.precert_entry();
      for (int n = 0; n < precertchain.precertificate_chain_size(); ++n)
        WriteCertificate(precertchain.precertificate_chain(n), e, n + 1,
                         "x509");
    }
  }
}

int GetRoots() {
  HTTPLogClient client(FLAGS_ct_server);

  vector<unique_ptr<Cert>> roots;
  CHECK_EQ(client.GetRoots(&roots), AsyncLogClient::OK);

  LOG(INFO) << "number of certs: " << roots.size();
  for (vector<unique_ptr<Cert>>::const_iterator it = roots.begin();
       it != roots.end(); ++it) {
    string pem_cert;
    CHECK_EQ((*it)->PemEncoding(&pem_cert), util::Status::OK);
    std::cout << pem_cert;
  }

  std::cout << std::endl;

  return 0;
}

int GetSTH() {
  CHECK_NE(FLAGS_ct_server, "");

  HTTPLogClient client(FLAGS_ct_server);

  ct::SignedTreeHead sth;
  CHECK_EQ(AsyncLogClient::OK, client.GetSTH(&sth));

  const unique_ptr<LogVerifier> verifier(GetLogVerifierFromFlags());

  // Allow for 10 seconds of clock skew
  uint64_t latest = ((uint64_t)time(NULL) + 10) * 1000;
  const LogVerifier::LogVerifyResult result =
      verifier->VerifySignedTreeHead(sth, 0, latest);

  LOG(INFO) << "STH is " << sth.DebugString();

  if (result != LogVerifier::VERIFY_OK) {
    if (result == LogVerifier::INVALID_TIMESTAMP)
      LOG(ERROR) << "STH has bad timestamp (" << sth.timestamp() << ")";
    else if (result == LogVerifier::INVALID_SIGNATURE)
      LOG(ERROR) << "STH signature doesn't validate";
    else
      LOG(ERROR) << "STH validation failed with unknown error " << result;
    return 1;
  }

  string sth_str;
  CHECK(sth.SerializeToString(&sth_str));
  WriteFile(FLAGS_ct_server_response_out, sth_str, "STH");

  return 0;
}

static monitor::Database* GetMonitorDBFromFlags() {
  CHECK_NE(FLAGS_sqlite_db, "");
  monitor::Database* db;
  db = new monitor::SQLiteDB(FLAGS_sqlite_db);
  return db;
}

// Return code 0 indicates success.
// See monitor class for the monitor action specific return codes.
int Monitor() {
  CHECK_NE(FLAGS_monitor_action, "");
  CHECK_NE(FLAGS_ct_server, "");

  HTTPLogClient client(FLAGS_ct_server);
  monitor::Monitor monitor(GetMonitorDBFromFlags(), GetLogVerifierFromFlags(),
                           &client, FLAGS_monitor_sleep_time_secs);

  int ret = 0;
  if (FLAGS_monitor_action == "get_sth") {
    ret = monitor.GetSTH();
  } else if (FLAGS_monitor_action == "verify_sth") {
    ret = monitor.VerifySTH(FLAGS_timestamp);
  } else if (FLAGS_monitor_action == "get_entries") {
    ret = monitor.GetEntries(FLAGS_get_first, FLAGS_get_last);
  } else if (FLAGS_monitor_action == "confirm_tree") {
    ret = monitor.ConfirmTree(FLAGS_timestamp);
  } else if (FLAGS_monitor_action == "init") {
    monitor.Init();
  } else if (FLAGS_monitor_action == "loop") {
    monitor.Loop();
  } else {
    LOG(FATAL) << "Wrong monitor_action flag given.";
  }
  return ret;
}


// Exit code upon normal exit:
// 0: success
// 1: failure
// - for log server: connection failed or the server replied with an error
// - for SSL server: connection failed, handshake failed when success was
//                   expected or vice versa
// 2: initial connection to the (log/ssl) server failed
// Exit code upon abnormal exit (CHECK failures): != 0
// (on UNIX, 134 is expected)
int main(int argc, char** argv) {
  google::SetUsageMessage(argv[0] + string(kUsage));
  util::InitCT(&argc, &argv);

  const string main_command(argv[0]);
  if (argc < 2) {
    std::cout << google::ProgramUsage();
    return 1;
  }

  const string cmd(argv[1]);

  int ret = 0;
  if (cmd == "connect") {
    bool want_fail = FLAGS_ssl_client_expect_handshake_failure;
    SSLClient::HandshakeResult result = Connect();
    if ((!want_fail && result != SSLClient::OK) ||
        (want_fail && result != SSLClient::HANDSHAKE_FAILED))
      ret = 1;
  } else if (cmd == "upload") {
    ret = Upload();
  } else if (cmd == "audit") {
    ret = Audit();
  } else if (cmd == "consistency") {
    ret = CheckConsistency();
  } else if (cmd == "certificate") {
    MakeCert();
  } else if (cmd == "extension_data") {
    ProofToExtensionData();
  } else if (cmd == "configure_proof") {
    WriteProofToConfig();
  } else if (cmd == "diagnose_chain") {
    DiagnoseCertChain();
  } else if (cmd == "wrap") {
    Wrap();
  } else if (cmd == "wrap_embedded") {
    WrapEmbedded();
  } else if (cmd == "get_entries") {
    GetEntries();
  } else if (cmd == "get_roots") {
    ret = GetRoots();
  } else if (cmd == "monitor") {
    ret = Monitor();
  } else if (cmd == "sth") {
    ret = GetSTH();
  } else {
    std::cout << google::ProgramUsage();
    ret = 1;
  }

  return ret;
}
