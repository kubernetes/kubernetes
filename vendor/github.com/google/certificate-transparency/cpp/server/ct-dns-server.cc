#include <gflags/gflags.h>
#include <ldns/ldns.h>
#include <iostream>
#include <sstream>
#include <string>

#include "log/log_lookup.h"
#include "log/logged_entry.h"
#include "log/sqlite_db.h"
#include "proto/cert_serializer.h"
#include "proto/ct.pb.h"
#include "server/event.h"
#include "util/init.h"
#include "util/util.h"

using cert_trans::LogLookup;
using cert_trans::LoggedEntry;
using cert_trans::SQLiteDB;
using ct::SignedTreeHead;
using google::RegisterFlagValidator;
using std::string;
using std::stringstream;

DEFINE_int32(port, 0, "Server port");
DEFINE_string(domain, "", "Domain");
DEFINE_string(db, "", "Database for certificate and tree storage");

// Basic sanity checks on flag values.
static bool ValidatePort(const char*, int32_t port) {
  return (port <= 0 || port > 65535);
}

static const bool port_dummy =
    RegisterFlagValidator(&FLAGS_port, &ValidatePort);

static bool NonEmptyString(const char*, const string& str) {
  return !str.empty();
}

static const bool domain_dummy =
    RegisterFlagValidator(&FLAGS_domain, &NonEmptyString);

class CTUDPDNSServer : public UDPServer {
 public:
  CTUDPDNSServer(const string& domain, SQLiteDB* db, EventLoop* loop, int fd)
      : UDPServer(loop, fd), domain_(domain), lookup_(db), db_(db) {
  }

  virtual void PacketRead(const sockaddr_in& from, const char* buf,
                          size_t len) {
    ldns_pkt* packet = NULL;

    ldns_status ret = ldns_wire2pkt(&packet, (const uint8_t*)buf, len);
    if (ret != LDNS_STATUS_OK) {
      LOG(INFO) << "Bad DNS packet";
      return;
    }

    // ldns_pkt_print(stdout, packet);

    if (ldns_pkt_qr(packet) != 0) {
      LOG(INFO) << "Packet is not a query";
      return;
    }

    if (ldns_pkt_get_opcode(packet) != LDNS_PACKET_QUERY) {
      LOG(INFO) << "Packet has bad opcode";
      return;
    }

    ldns_pkt* answers = ldns_pkt_new();
    ldns_pkt_set_id(answers, ldns_pkt_id(packet));
    ldns_pkt_set_qr(answers, true);

    ldns_rr_list* questions = ldns_pkt_question(packet);

    ldns_pkt_safe_push_rr_list(answers, LDNS_SECTION_QUESTION,
                               ldns_rr_list_clone(questions));

    for (size_t n = 0; n < ldns_rr_list_rr_count(questions); ++n) {
      ldns_rr* question = ldns_rr_list_rr(questions, n);

      if (ldns_rr_get_type(question) != LDNS_RR_TYPE_TXT) {
        LOG(INFO) << "Question is not TXT";
        // FIXME(benl): set error response?
        continue;
      }

      ldns_rdf* owner = ldns_rr_owner(question);
      if (ldns_rdf_get_type(owner) != LDNS_RDF_TYPE_DNAME) {
        LOG(INFO) << "Owner is not a dname";
        continue;
      }

      ldns_buffer* dname = ldns_buffer_new(512);
      if (ldns_rdf2buffer_str_dname(dname, owner) != LDNS_STATUS_OK) {
        LOG(INFO) << "Can't decode owner";
        continue;
      }

      char* owner_name_raw = ldns_buffer2str(dname);
      std::string owner_name(owner_name_raw);
      free(owner_name_raw);
      owner_name_raw = NULL;
      ldns_buffer_free(dname);
      dname = NULL;

      LOG(INFO) << "Question is TXT of " << owner_name;

      if (owner_name.length() <= domain_.length() ||
          owner_name.compare(owner_name.length() - domain_.length(),
                             domain_.length(), domain_) != 0) {
        LOG(INFO) << "Question is not for our domain";
        continue;
      }

      std::string response = Response(
          owner_name.substr(0, owner_name.length() - domain_.length() - 1));

      ldns_rr* answer = ldns_rr_new();
      ldns_rr_set_owner(answer, ldns_rdf_new_frm_str(LDNS_RDF_TYPE_DNAME,
                                                     owner_name.c_str()));
      ldns_rr_set_type(answer, LDNS_RR_TYPE_TXT);
      ldns_rr_set_ttl(answer, 123);
      ldns_rr_push_rdf(answer, ldns_rdf_new_frm_str(LDNS_RDF_TYPE_STR,
                                                    response.c_str()));
      ldns_pkt_safe_push_rr(answers, LDNS_SECTION_ANSWER, answer);
    }
    ldns_pkt_free(packet);

    char* answer_str = ldns_pkt2str(answers);
    LOG(INFO) << "Answer is " << answer_str;
    free(answer_str);

    uint8_t* wire_answer;
    size_t answer_size;
    if (ldns_pkt2wire(&wire_answer, answers, &answer_size) != LDNS_STATUS_OK) {
      LOG(ERROR) << "Can't make wire answer";
      return;
    }
    QueuePacket(from, wire_answer, answer_size);
    free(wire_answer);
    ldns_pkt_free(answers);
  }

 private:
  string Response(string question) {
    if (question == "sth")
      return STH();

    size_t dot = question.find_last_of('.');
    if (dot == string::npos)
      return question + " not understood";

    string head = question.substr(0, dot);
    string tail = question.substr(dot + 1);
    LOG(INFO) << "head = " << head << ", tail = " << tail;
    if (tail == "tree")
      return Tree(head);
    else if (tail == "hash")
      return Hash(head);
    else if (tail == "leafhash")
      return LeafHash(head);

    return question + " is the question.";
  }

  string LeafHash(const string& index_str) const {
    int index = atoi(index_str.c_str());
    LoggedEntry cert;
    if (db_->LookupByIndex(index, &cert) != db_->LOOKUP_OK)
      return "No such index";
    return util::ToBase64(lookup_.LeafHash(cert));
  }

  string Hash(const string& hash) {
    db_->ForceNotifySTH();

    // FIXME: decode hash!
    int64_t index;
    if (lookup_.GetIndex(hash, &index) != lookup_.OK)
      return "No such hash";

    stringstream ss;
    ss << index;
    return ss.str();
  }

  string Tree(const string& question) {
    size_t dot = question.find_first_of('.');
    if (dot == string::npos)
      return question + " not understood";

    size_t dot2 = question.find_first_of('.', dot + 1);
    if (dot2 == string::npos)
      return question + " not understood";

    string level = question.substr(0, dot);
    string index = question.substr(dot + 1, dot2 - dot - 1);
    string size = question.substr(dot2 + 1);

    LOG(INFO) << "level = " << level << ", index = " << index
              << ", size = " << size;

    ct::ShortMerkleAuditProof proof;
    if (lookup_.AuditProof(atoi(index.c_str()), atoi(size.c_str()), &proof) !=
        lookup_.OK)
      return "Lookup of node " + index + "." + size + " failed";

    int l = atoi(level.c_str());
    if (l < 0 || l >= proof.path_node_size())
      return "Level " + level + " is out of range";

    string b64 = util::ToBase64(proof.path_node(l));
    return b64;
  }

  string STH() {
    db_->ForceNotifySTH();

    const SignedTreeHead& sth = lookup_.GetSTH();

    std::string signature;
    CHECK_EQ(Serializer::SerializeDigitallySigned(sth.signature(), &signature),
             SerializeResult::OK);

    stringstream ss;
    ss << sth.tree_size() << '.' << sth.timestamp() << '.'
       << util::ToBase64(sth.sha256_root_hash()) << '.'
       << util::ToBase64(signature);

    return ss.str();
  }

  string domain_;
  LogLookup lookup_;
  SQLiteDB* const db_;
};

class Keyboard : public Server {
 public:
  Keyboard(EventLoop* loop) : Server(loop, 0) {
  }

  void BytesRead(std::string* rbuffer) {
    while (!rbuffer->empty()) {
      ProcessKey(rbuffer->at(0));
      rbuffer->erase(0, 1);
    }
  }

 private:
  void ProcessKey(char key) {
    switch (key) {
      case 'q':
        loop()->Stop();
        break;

      case '\n':
        break;

      default:
        std::cout << "Don't understand " << key << std::endl;
        break;
    }
  }
};

int main(int argc, char* argv[]) {
  util::InitCT(&argc, &argv);
  ConfigureSerializerForV1CT();

  // TODO(pphaneuf): This current *has* to be SQLite, because it
  // depends on sharing the database with a ct-server that will
  // populate it (which FileDB does not support).
  SQLiteDB db(FLAGS_db);

  EventLoop loop;

  // Mostly so we can have a clean exit for valgrind etc.
  Keyboard keyboard(&loop);

  int dns_fd;
  CHECK(Services::InitServer(&dns_fd, FLAGS_port, NULL, SOCK_DGRAM));
  CTUDPDNSServer dns(FLAGS_domain, &db, &loop, dns_fd);

  LOG(INFO) << "Server listening on port " << FLAGS_port;
  loop.Forever();
}
