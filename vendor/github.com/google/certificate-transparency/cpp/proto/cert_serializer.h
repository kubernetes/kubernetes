/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef CERT_TRANS_PROTO_CERT_SERIALIZER_H_
#define CERT_TRANS_PROTO_CERT_SERIALIZER_H_

#include <glog/logging.h>
#include <google/protobuf/repeated_field.h>
#include <stdint.h>
#include <string>

#include "base/macros.h"
#include "proto/ct.pb.h"
#include "proto/serializer.h"


void ConfigureSerializerForV1CT();
void ConfigureSerializerForV2CT();

// NB This serializes the certificate_chain component of the X509 chain only.
// Needed for the GetEntries flow.
SerializeResult SerializeX509Chain(const ct::X509ChainEntry& entry,
                                   std::string* result);

SerializeResult SerializeX509ChainV1(const repeated_string& certificate_chain,
                                     std::string* result);

SerializeResult SerializePrecertChainEntry(const ct::PrecertChainEntry& entry,
                                           std::string* result);

SerializeResult SerializePrecertChainEntry(
    const std::string& pre_certificate,
    const repeated_string& precertificate_chain, std::string* result);

// These two functions are depended on externally.
SerializeResult SerializeV1SignedCertEntryWithType(
    const std::string& leaf_certificate, std::string* result);

SerializeResult SerializeV1SignedPrecertEntryWithType(
    const std::string& issuer_key_hash, const std::string& tbs_certificate,
    std::string* result);

DeserializeResult DeserializeX509Chain(const std::string& in,
                                       ct::X509ChainEntry* x509_chain_entry);

DeserializeResult DeserializePrecertChainEntry(
    const std::string& in, ct::PrecertChainEntry* precert_chain_entry);

// Test helpers
//
SerializeResult SerializeV1CertSCTMerkleTreeLeaf(
    uint64_t timestamp, const std::string& certificate,
    const std::string& extensions, std::string* result);

SerializeResult SerializeV1PrecertSCTMerkleTreeLeaf(
    uint64_t timestamp, const std::string& issuer_key_hash,
    const std::string& tbs_certificate, const std::string& extensions,
    std::string* result);

SerializeResult SerializeV2CertSCTMerkleTreeLeaf(
    uint64_t timestamp, const std::string& issuer_key_hash,
    const std::string& tbs_certificate,
    const google::protobuf::RepeatedPtrField<ct::SctExtension>& sct_extension,
    std::string* result);

SerializeResult SerializeV2PrecertSCTMerkleTreeLeaf(
    uint64_t timestamp, const std::string& issuer_key_hash,
    const std::string& tbs_certificate,
    const google::protobuf::RepeatedPtrField<ct::SctExtension>& sct_extension,
    std::string* result);

SerializeResult SerializeV2CertSCTSignatureInput(
    uint64_t timestamp, const std::string& issuer_key_hash,
    const std::string& tbs_certificate,
    const google::protobuf::RepeatedPtrField<ct::SctExtension>& sct_extension,
    std::string* result);

SerializeResult SerializeV1CertSCTSignatureInput(
    uint64_t timestamp, const std::string& certificate,
    const std::string& extensions, std::string* result);

SerializeResult SerializeV1PrecertSCTSignatureInput(
    uint64_t timestamp, const std::string& issuer_key_hash,
    const std::string& tbs_certificate, const std::string& extensions,
    std::string* result);

SerializeResult SerializeV2PrecertSCTSignatureInput(
    uint64_t timestamp, const std::string& issuer_key_hash,
    const std::string& tbs_certificate,
    const google::protobuf::RepeatedPtrField<ct::SctExtension>& sct_extension,
    std::string* result);

#endif  // CERT_TRANS_PROTO_CERT_SERIALIZER_H_
