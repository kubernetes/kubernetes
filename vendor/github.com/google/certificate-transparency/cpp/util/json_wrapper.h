/* -*- mode: c++; indent-tabs-mode: nil -*- */
#ifndef JSON_WRAPPER_H
#define JSON_WRAPPER_H

#include <glog/logging.h>
#include <json.h>
#undef TRUE   // json.h pollution
#undef FALSE  // json.h pollution

#include <event2/buffer.h>
#include <sstream>
#include <string>

#include "base/macros.h"
#include "proto/serializer.h"
#include "util/util.h"

class JsonArray;

// It appears that a new object, e.g. from a string, has a reference count
// of 1, and that any objects "got" from it will get freed when it is freed.

// Note that a JsonObject that is not Ok() should not be used for anything.
class JsonObject {
 public:
  explicit JsonObject(json_object* obj) : obj_(obj) {
  }

  explicit JsonObject(const std::ostringstream& response) {
    obj_ = json_tokener_parse(response.str().c_str());
  }

  explicit JsonObject(const std::string& response) {
    obj_ = json_tokener_parse(response.c_str());
  }

  // This constructor is destructive: if a JSON object is parsed
  // correctly, it will remove it from the front of the buffer. In
  // case of an error, the buffer is left unchanged.
  explicit JsonObject(evbuffer* buffer);

  JsonObject(const JsonArray& from, int offset,
             json_type type = json_type_object);

  JsonObject(const JsonObject& from, const char* field) {
    InitFromChild(from, field, json_type_object);
  }

  JsonObject() : obj_(json_object_new_object()) {
  }

  ~JsonObject() {
    if (obj_)
      json_object_put(obj_);
  }

  // Get the object out, and stop tracking it so we _won't_ put() it
  // when we are destroyed. The caller needs to ensure it is freed.
  json_object* Extract() {
    json_object* tmp = obj_;
    obj_ = NULL;
    return tmp;
  }

  bool Ok() const {
    return obj_ != NULL;
  }

  bool IsType(json_type type) const {
    return json_object_is_type(obj_, type);
  }

  const char* ToJson() const {
    return json_object_to_json_string(obj_);
  }

  void Add(const char* name, const JsonObject& addand) {
    Add(name, json_object_get(addand.obj_));
  }

  void Add(const char* name, int64_t value) {
    Add(name, json_object_new_int64(value));
  }

  void Add(const char* name, const std::string& value) {
    Add(name, json_object_new_string(value.c_str()));
  }

  void AddBase64(const char* name, const std::string& value) {
    Add(name, util::ToBase64(value));
  }

  void Add(const char* name, const ct::DigitallySigned& ds) {
    std::string signature;
    CHECK_EQ(Serializer::SerializeDigitallySigned(ds, &signature),
             SerializeResult::OK);
    AddBase64(name, signature);
  }

  void AddBoolean(const char* name, bool b) {
    Add(name, json_object_new_boolean(b));
  }

  const char* ToString() const {
    return json_object_to_json_string(obj_);
  }

  std::string DebugString() const {
    return json_object_to_json_string_ext(obj_, JSON_C_TO_STRING_PRETTY);
  }

 protected:
  JsonObject(const JsonObject& from, const char* field, json_type type) {
    InitFromChild(from, field, type);
  }

  json_object* obj_;

 private:
  void InitFromChild(const JsonObject& from, const char* field,
                     json_type type) {
    if (json_object_object_get_ex(from.obj_, field, &obj_)) {
      if (!json_object_is_type(obj_, type)) {
        LOG(ERROR) << "Don't understand " << field
                   << " field: " << from.ToJson();
        obj_ = NULL;
        return;
      }
    } else {
      VLOG(2) << "No " << field << " field";
      return;
    }
    // Increment reference count
    json_object_get(obj_);
  }

  void Add(const char* name, json_object* obj) {
    json_object_object_add(obj_, name, obj);
  }

  DISALLOW_COPY_AND_ASSIGN(JsonObject);
};

class JsonBoolean : public JsonObject {
 public:
  JsonBoolean(const JsonObject& from, const char* field)
      : JsonObject(from, field, json_type_boolean) {
  }

  bool Value() const {
    return json_object_get_boolean(obj_);
  }
};

class JsonString : public JsonObject {
 public:
  JsonString(const JsonObject& from, const char* field)
      : JsonObject(from, field, json_type_string) {
  }

  JsonString(const JsonArray& from, int offset)
      : JsonObject(from, offset, json_type_string) {
  }

  const char* Value() const {
    return json_object_get_string(obj_);
  }

  std::string FromBase64() {
    return util::FromBase64(Value());
  }
};

class JsonInt : public JsonObject {
 public:
  explicit JsonInt(json_object* jint) : JsonObject(jint) {
  }
  JsonInt(const JsonObject& from, const char* field)
      : JsonObject(from, field, json_type_int) {
  }

  int64_t Value() const {
    return json_object_get_int64(obj_);
  }
};

class JsonArray : public JsonObject {
 public:
  JsonArray(const JsonObject& from, const char* field)
      : JsonObject(from, field, json_type_array) {
  }

  JsonArray() : JsonObject(json_object_new_array()) {
  }

  void Add(json_object* addand) {
    json_object_array_add(obj_, addand);
  }

  void Add(const std::string& addand) {
    Add(json_object_new_string(addand.c_str()));
  }

  void Add(JsonObject* addand) {
    Add(addand->Extract());
  }

  void AddBase64(const std::string& addand) {
    Add(util::ToBase64(addand));
  }

  int Length() const {
    return json_object_array_length(obj_);
  }
};

#endif
