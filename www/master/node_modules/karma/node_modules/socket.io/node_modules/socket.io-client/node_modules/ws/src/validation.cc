/*!
 * ws: a node.js websocket client
 * Copyright(c) 2011 Einar Otto Stangvik <einaros@gmail.com>
 * MIT Licensed
 */

#include <v8.h>
#include <node.h>
#include <node_buffer.h>
#include <node_object_wrap.h>
#include <stdlib.h>
#include <wchar.h>
#include <stdio.h>
#include "nan.h"

using namespace v8;
using namespace node;

#define UNI_SUR_HIGH_START   (uint32_t) 0xD800
#define UNI_SUR_LOW_END    (uint32_t) 0xDFFF
#define UNI_REPLACEMENT_CHAR (uint32_t) 0x0000FFFD
#define UNI_MAX_LEGAL_UTF32  (uint32_t) 0x0010FFFF

static const uint8_t trailingBytesForUTF8[256] = {
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
  1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,4,4,4,4,5,5,5,5
};

static const uint32_t offsetsFromUTF8[6] = {
  0x00000000, 0x00003080, 0x000E2080,
  0x03C82080, 0xFA082080, 0x82082080
};

static int isLegalUTF8(const uint8_t *source, const int length)
{
  uint8_t a;
  const uint8_t *srcptr = source+length;
  switch (length) {
  default: return 0;
  /* Everything else falls through when "true"... */
  /* RFC3629 makes 5 & 6 bytes UTF-8 illegal
  case 6: if ((a = (*--srcptr)) < 0x80 || a > 0xBF) return 0;
  case 5: if ((a = (*--srcptr)) < 0x80 || a > 0xBF) return 0; */
  case 4: if ((a = (*--srcptr)) < 0x80 || a > 0xBF) return 0;
  case 3: if ((a = (*--srcptr)) < 0x80 || a > 0xBF) return 0;
  case 2: if ((a = (*--srcptr)) > 0xBF) return 0;
    switch (*source) {
      /* no fall-through in this inner switch */
      case 0xE0: if (a < 0xA0) return 0; break;
      case 0xED: if (a > 0x9F) return 0; break;
      case 0xF0: if (a < 0x90) return 0; break;
      case 0xF4: if (a > 0x8F) return 0; break;
      default:   if (a < 0x80) return 0;
    }

  case 1: if (*source >= 0x80 && *source < 0xC2) return 0;
  }
  if (*source > 0xF4) return 0;
  return 1;
}

int is_valid_utf8 (size_t len, char *value)
{
  /* is the string valid UTF-8? */
  for (unsigned int i = 0; i < len; i++) {
    uint32_t ch = 0;
    uint8_t  extrabytes = trailingBytesForUTF8[(uint8_t) value[i]];

    if (extrabytes + i >= len)
      return 0;

    if (isLegalUTF8 ((uint8_t *) (value + i), extrabytes + 1) == 0) return 0;

    switch (extrabytes) {
      case 5 : ch += (uint8_t) value[i++]; ch <<= 6;
      case 4 : ch += (uint8_t) value[i++]; ch <<= 6;
      case 3 : ch += (uint8_t) value[i++]; ch <<= 6;
      case 2 : ch += (uint8_t) value[i++]; ch <<= 6;
      case 1 : ch += (uint8_t) value[i++]; ch <<= 6;
      case 0 : ch += (uint8_t) value[i];
    }

    ch -= offsetsFromUTF8[extrabytes];

    if (ch <= UNI_MAX_LEGAL_UTF32) {
      if (ch >= UNI_SUR_HIGH_START && ch <= UNI_SUR_LOW_END)
        return 0;
    } else {
      return 0;
    }
  }

  return 1;
}

class Validation : public ObjectWrap
{
public:

  static void Initialize(v8::Handle<v8::Object> target)
  {
    NanScope();
    Local<FunctionTemplate> t = NanNew<FunctionTemplate>(New);
    t->InstanceTemplate()->SetInternalFieldCount(1);
    NODE_SET_METHOD(t, "isValidUTF8", Validation::IsValidUTF8);
    target->Set(NanSymbol("Validation"), t->GetFunction());
  }

protected:

  static NAN_METHOD(New)
  {
    NanScope();
    Validation* validation = new Validation();
    validation->Wrap(args.This());
    NanReturnValue(args.This());
  }

  static NAN_METHOD(IsValidUTF8)
  {
    NanScope();
    if (!Buffer::HasInstance(args[0])) {
      return NanThrowTypeError("First argument needs to be a buffer");
    }
    Local<Object> buffer_obj = args[0]->ToObject();
    char *buffer_data = Buffer::Data(buffer_obj);
    size_t buffer_length = Buffer::Length(buffer_obj);
    NanReturnValue(is_valid_utf8(buffer_length, buffer_data) == 1 ? NanTrue() : NanFalse());
  }
};

extern "C" void init (Handle<Object> target)
{
  NanScope();
  Validation::Initialize(target);
}

NODE_MODULE(validation, init)

