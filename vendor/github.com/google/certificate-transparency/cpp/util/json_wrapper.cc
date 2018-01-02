#include "json_wrapper.h"

#include <memory>

using std::unique_ptr;


JsonObject::JsonObject(evbuffer* buffer) : obj_(NULL) {
  const unique_ptr<json_tokener, void (*)(json_tokener*)> tokener(
      json_tokener_new(), json_tokener_free);

  evbuffer_ptr ptr;
  evbuffer_ptr_set(buffer, &ptr, 0, EVBUFFER_PTR_SET);

  unsigned amount_consumed(0);

  while (!obj_ && amount_consumed < evbuffer_get_length(buffer)) {
    evbuffer_iovec chunk;

    if (evbuffer_peek(buffer, -1, &ptr, &chunk, 1) < 1) {
      // No more data.
      break;
    }

    // This function can be called repeatedly with each chunk of
    // data. It keeps its state in "*tokener", and will return a
    // non-NULL value once it finds a full object. If it returns NULL
    // and the error is "json_tokener_continue", this simply means
    // that it hasn't yet found an object, we just need to keep
    // calling it with more data.
    obj_ = json_tokener_parse_ex(tokener.get(),
                                 static_cast<char*>(chunk.iov_base),
                                 chunk.iov_len);

    // Check for a parsing error.
    if (!obj_ &&
        json_tokener_get_error(tokener.get()) != json_tokener_continue) {
      VLOG(1) << "json_tokener_parse_ex: "
              << json_tokener_error_desc(
                     json_tokener_get_error(tokener.get()));
      break;
    }

    if (obj_) {
      // At the end of the parsing, we might not have consumed all the
      // bytes in the iovec.
      amount_consumed += tokener->char_offset;
      // No need to update "ptr" here, we're done.
    } else {
      amount_consumed += chunk.iov_len;
      evbuffer_ptr_set(buffer, &ptr, chunk.iov_len, EVBUFFER_PTR_ADD);
    }
  }

  // If the parsing was successful, drain the number of bytes
  // consumed.
  if (obj_) {
    evbuffer_drain(buffer, amount_consumed);
  }
}


JsonObject::JsonObject(const JsonArray& from, int offset, json_type type) {
  obj_ = json_object_array_get_idx(from.obj_, offset);
  if (obj_ != NULL) {
    if (!json_object_is_type(obj_, type)) {
      LOG(ERROR) << "Don't understand index " << offset << ": "
                 << from.ToJson();
      obj_ = NULL;
      return;
    }
  } else {
    LOG(ERROR) << "No index " << offset;
    return;
  }
  json_object_get(obj_);
}
