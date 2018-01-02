/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef CERT_TRANS_PROTO_XJSON_SERIALIZER_H_
#define CERT_TRANS_PROTO_XJSON_SERIALIZER_H_

#include <glog/logging.h>
#include <google/protobuf/repeated_field.h>
#include <stdint.h>
#include <string>

#include "base/macros.h"
#include "proto/ct.pb.h"


void ConfigureSerializerForV1XJSON();


#endif  // CERT_TRANS_PROTO_XJSON_SERIALIZER_H_
