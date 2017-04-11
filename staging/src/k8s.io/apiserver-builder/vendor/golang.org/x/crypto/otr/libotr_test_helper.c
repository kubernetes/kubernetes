// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code can be compiled and used to test the otr package against libotr.
// See otr_test.go.

// +build ignore

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <proto.h>
#include <message.h>
#include <privkey.h>

static int g_session_established = 0;

OtrlPolicy policy(void *opdata, ConnContext *context) {
  return OTRL_POLICY_ALWAYS;
}

int is_logged_in(void *opdata, const char *accountname, const char *protocol,
                 const char *recipient) {
  return 1;
}

void inject_message(void *opdata, const char *accountname, const char *protocol,
                    const char *recipient, const char *message) {
  printf("%s\n", message);
  fflush(stdout);
  fprintf(stderr, "libotr helper sent: %s\n", message);
}

void update_context_list(void *opdata) {}

void new_fingerprint(void *opdata, OtrlUserState us, const char *accountname,
                     const char *protocol, const char *username,
                     unsigned char fingerprint[20]) {
  fprintf(stderr, "NEW FINGERPRINT\n");
  g_session_established = 1;
}

void write_fingerprints(void *opdata) {}

void gone_secure(void *opdata, ConnContext *context) {}

void gone_insecure(void *opdata, ConnContext *context) {}

void still_secure(void *opdata, ConnContext *context, int is_reply) {}

int max_message_size(void *opdata, ConnContext *context) { return 99999; }

const char *account_name(void *opdata, const char *account,
                         const char *protocol) {
  return "ACCOUNT";
}

void account_name_free(void *opdata, const char *account_name) {}

const char *error_message(void *opdata, ConnContext *context,
                          OtrlErrorCode err_code) {
  return "ERR";
}

void error_message_free(void *opdata, const char *msg) {}

void resent_msg_prefix_free(void *opdata, const char *prefix) {}

void handle_smp_event(void *opdata, OtrlSMPEvent smp_event,
                      ConnContext *context, unsigned short progress_event,
                      char *question) {}

void handle_msg_event(void *opdata, OtrlMessageEvent msg_event,
                      ConnContext *context, const char *message,
                      gcry_error_t err) {
  fprintf(stderr, "msg event: %d %s\n", msg_event, message);
}

OtrlMessageAppOps uiops = {
    policy,
    NULL,
    is_logged_in,
    inject_message,
    update_context_list,
    new_fingerprint,
    write_fingerprints,
    gone_secure,
    gone_insecure,
    still_secure,
    max_message_size,
    account_name,
    account_name_free,
    NULL, /* received_symkey */
    error_message,
    error_message_free,
    NULL, /* resent_msg_prefix */
    resent_msg_prefix_free,
    handle_smp_event,
    handle_msg_event,
    NULL /* create_instag */,
    NULL /* convert_msg */,
    NULL /* convert_free */,
    NULL /* timer_control */,
};

static const char kPrivateKeyData[] =
    "(privkeys (account (name \"account\") (protocol proto) (private-key (dsa "
    "(p "
    "#00FC07ABCF0DC916AFF6E9AE47BEF60C7AB9B4D6B2469E436630E36F8A489BE812486A09F"
    "30B71224508654940A835301ACC525A4FF133FC152CC53DCC59D65C30A54F1993FE13FE63E"
    "5823D4C746DB21B90F9B9C00B49EC7404AB1D929BA7FBA12F2E45C6E0A651689750E8528AB"
    "8C031D3561FECEE72EBB4A090D450A9B7A857#) (q "
    "#00997BD266EF7B1F60A5C23F3A741F2AEFD07A2081#) (g "
    "#535E360E8A95EBA46A4F7DE50AD6E9B2A6DB785A66B64EB9F20338D2A3E8FB0E94725848F"
    "1AA6CC567CB83A1CC517EC806F2E92EAE71457E80B2210A189B91250779434B41FC8A8873F"
    "6DB94BEA7D177F5D59E7E114EE10A49CFD9CEF88AE43387023B672927BA74B04EB6BBB5E57"
    "597766A2F9CE3857D7ACE3E1E3BC1FC6F26#) (y "
    "#0AC8670AD767D7A8D9D14CC1AC6744CD7D76F993B77FFD9E39DF01E5A6536EF65E775FCEF"
    "2A983E2A19BD6415500F6979715D9FD1257E1FE2B6F5E1E74B333079E7C880D39868462A93"
    "454B41877BE62E5EF0A041C2EE9C9E76BD1E12AE25D9628DECB097025DD625EF49C3258A1A"
    "3C0FF501E3DC673B76D7BABF349009B6ECF#) (x "
    "#14D0345A3562C480A039E3C72764F72D79043216#)))))\n";

int main() {
  OTRL_INIT;

  // We have to write the private key information to a file because the libotr
  // API demands a filename to read from.
  const char *tmpdir = "/tmp";
  if (getenv("TMP")) {
    tmpdir = getenv("TMP");
  }

  char private_key_file[256];
  snprintf(private_key_file, sizeof(private_key_file),
           "%s/libotr_test_helper_privatekeys-XXXXXX", tmpdir);
  int fd = mkstemp(private_key_file);
  if (fd == -1) {
    perror("creating temp file");
  }
  write(fd, kPrivateKeyData, sizeof(kPrivateKeyData) - 1);
  close(fd);

  OtrlUserState userstate = otrl_userstate_create();
  otrl_privkey_read(userstate, private_key_file);
  unlink(private_key_file);

  fprintf(stderr, "libotr helper started\n");

  char buf[4096];

  for (;;) {
    char *message = fgets(buf, sizeof(buf), stdin);
    if (strlen(message) == 0) {
      break;
    }
    message[strlen(message) - 1] = 0;
    fprintf(stderr, "libotr helper got: %s\n", message);

    char *newmessage = NULL;
    OtrlTLV *tlvs;
    int ignore_message = otrl_message_receiving(
        userstate, &uiops, NULL, "account", "proto", "peer", message,
        &newmessage, &tlvs, NULL, NULL, NULL);
    if (tlvs) {
      otrl_tlv_free(tlvs);
    }

    if (newmessage != NULL) {
      fprintf(stderr, "libotr got: %s\n", newmessage);
      otrl_message_free(newmessage);

      gcry_error_t err;
      char *newmessage = NULL;

      err = otrl_message_sending(userstate, &uiops, NULL, "account", "proto",
                                 "peer", 0, "test message", NULL, &newmessage,
                                 OTRL_FRAGMENT_SEND_SKIP, NULL, NULL, NULL);
      if (newmessage == NULL) {
        fprintf(stderr, "libotr didn't encrypt message\n");
        return 1;
      }
      write(1, newmessage, strlen(newmessage));
      write(1, "\n", 1);
      fprintf(stderr, "libotr sent: %s\n", newmessage);
      otrl_message_free(newmessage);

      g_session_established = 0;
      write(1, "?OTRv2?\n", 8);
      fprintf(stderr, "libotr sent: ?OTRv2\n");
    }
  }

  return 0;
}
