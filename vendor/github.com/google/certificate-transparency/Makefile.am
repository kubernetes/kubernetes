ACLOCAL_AMFLAGS = -Im4

AM_CPPFLAGS = \
	-I$(GMOCK_DIR)/include \
	-I$(GTEST_DIR)/include \
	-Ithird_party/objecthash \
	$(evhtp_CFLAGS) \
	$(icu_CFLAGS) \
	$(json_c_CFLAGS)

AM_CXXFLAGS = \
	-fno-exceptions

BUILT_SOURCES = \
	cpp/gmock-all.cc \
	cpp/gtest-all.cc \
	cpp/monitoring/prometheus/metrics.pb.cc \
	cpp/monitoring/prometheus/metrics.pb.h \
	proto/ct.pb.cc \
	proto/ct.pb.h

BUILD_VERSION = `git describe --abbrev=0 --dirty --always --tags`

AM_TESTS_ENVIRONMENT = \
	LD_LIBRARY_PATH=$(INSTALL_DIR)/lib

# TODO(pphaneuf): change these to autoconf substitutions
export PROTOC
export INSTALL_DIR

.PHONY: cpp/version.cc
cpp/version.o: \
	CXXFLAGS+=-DBUILD_VERSION="\"$(BUILD_VERSION)\""

bin_PROGRAMS = \
	cpp/client/ct \
	cpp/server/ct-mirror \
	cpp/server/ct-mirror-v2 \
	cpp/server/ct-server \
	cpp/server/ct-server-v2 \
	cpp/server/xjson-server \
	cpp/tools/ct-clustertool

noinst_PROGRAMS = \
	cpp/tools/dump_cert \
	cpp/tools/dump_sth \
	cpp/tools/etcd_watch \
	cpp/tools/db_tool \
	cpp/util/bench_etcd \
	cpp/util/etcd_masterelection

if HAVE_LDNS
noinst_PROGRAMS += \
	cpp/server/ct-dns-server
endif

noinst_LIBRARIES = \
	cpp/libcore.a \
	cpp/libtest.a

check_PROGRAMS = \
	cpp/util/thread_pool_test \
	cpp/monitoring/gcm/exporter_test \
	cpp/net/url_fetcher_test \
	$(TESTS)

# TODO(pphaneuf): ct-dns-server-test.py has been disabled, because of
# commit 9391d114.
TESTS = \
	cpp/base/notification_test \
	cpp/fetcher/remote_peer_test \
	cpp/log/cert_checker_test \
	cpp/log/cert_submission_handler_test \
	cpp/log/cert_test \
	cpp/log/cluster_state_controller_test \
	cpp/log/cms_verifier_test \
	cpp/log/ct_extensions_test \
	cpp/log/database_large_test \
	cpp/log/database_test \
	cpp/log/etcd_consistent_store_test \
	cpp/log/file_storage_test \
	cpp/log/frontend_signer_test \
	cpp/log/frontend_test \
	cpp/log/log_lookup_test \
	cpp/log/log_signer_test \
	cpp/log/logged_entry_test \
	cpp/log/signer_verifier_test \
	cpp/log/strict_consistent_store_test \
	cpp/log/tree_signer_test \
	cpp/merkletree/merkle_tree_large_test \
	cpp/merkletree/merkle_tree_test \
	cpp/merkletree/serial_hasher_test \
	cpp/merkletree/sparse_merkle_tree_test \
	cpp/merkletree/tree_hasher_test \
	cpp/merkletree/verifiable_map_test \
	cpp/monitor/database_test \
	cpp/monitoring/counter_test \
	cpp/monitoring/gauge_test \
	cpp/monitoring/registry_test \
	cpp/proto/serializer_test \
	cpp/server/proxy_test \
	cpp/util/bignum_test \
	cpp/util/etcd_delete_test \
	cpp/util/etcd_test \
	cpp/util/fake_etcd_test \
	cpp/util/json_wrapper_test \
	cpp/util/libevent_wrapper_test \
	cpp/util/masterelection_test \
	cpp/util/sync_task_test \
	cpp/util/task_test

all-local:
	$(MAKE) -C python

clean-local:
	$(MAKE) -C python clean
if HAVE_ANT
	$(ANT) clean
endif

%.pb.h %.pb.cc: %.proto
	$(AM_V_GEN)$(PROTOC) $^ --cpp_out=.

cpp/gtest-all.cc: $(GTEST_DIR)/src/gtest-all.cc
	$(AM_V_at)cp $^ $@

cpp/gmock-all.cc: $(GMOCK_DIR)/src/gmock-all.cc
	$(AM_V_at)cp $^ $@

test/testdata/urlfetcher_test_certs/localhost-key.pem: test/create_url_fetcher_test_certs.sh
	$(AM_V_GEN)test/create_url_fetcher_test_certs.sh

cpp_libcore_a_SOURCES = \
	cpp/base/notification.cc \
	cpp/fetcher/continuous_fetcher.cc \
	cpp/fetcher/fetcher.cc \
	cpp/fetcher/peer.cc \
	cpp/fetcher/peer_group.cc \
	cpp/fetcher/remote_peer.cc \
	cpp/log/cert.cc \
	cpp/log/cert_checker.cc \
	cpp/log/cert_submission_handler.cc \
	cpp/log/cluster_state_controller_cert.cc \
	cpp/log/cms_verifier.cc \
	cpp/log/ct_extensions.cc \
	cpp/log/database.cc \
	cpp/log/etcd_consistent_store_cert.cc \
	cpp/log/file_db.cc \
	cpp/log/file_storage.cc \
	cpp/log/filesystem_ops.cc \
	cpp/log/frontend.cc \
	cpp/log/frontend_signer.cc \
	cpp/log/leveldb_db.cc \
	cpp/log/log_lookup.cc \
	cpp/log/log_signer.cc \
	cpp/log/log_verifier.cc \
	cpp/log/logged_entry.cc \
	cpp/log/signer.cc \
	cpp/log/sqlite_db.cc \
	cpp/log/strict_consistent_store_cert.cc \
	cpp/log/tree_signer_cert.cc \
	cpp/log/verifier.cc \
	cpp/merkletree/compact_merkle_tree.cc \
	cpp/merkletree/merkle_tree.cc \
	cpp/merkletree/merkle_tree_math.cc \
	cpp/merkletree/merkle_verifier.cc \
	cpp/merkletree/serial_hasher.cc \
	cpp/merkletree/sparse_merkle_tree.cc \
	cpp/merkletree/tree_hasher.cc \
	cpp/merkletree/verifiable_map.cc \
	cpp/monitoring/gcm/exporter.cc \
	cpp/monitoring/monitoring.cc \
	cpp/monitoring/prometheus/exporter.cc \
	cpp/monitoring/prometheus/metrics.pb.cc \
	cpp/monitoring/prometheus/metrics.pb.h \
	cpp/monitoring/registry.cc \
	cpp/net/connection_pool.cc \
	cpp/net/url.cc \
	cpp/net/url_fetcher.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/proto/xjson_serializer.cc \
	cpp/server/metrics.cc \
	cpp/server/proxy.cc \
	cpp/server/server.cc \
	cpp/server/staleness_tracker.cc \
	cpp/third_party/curl/hostcheck.c \
	cpp/third_party/isec_partners/openssl_hostname_validation.c \
	cpp/util/bignum.cc \
	cpp/util/etcd.cc \
	cpp/util/etcd_delete.cc \
	cpp/util/fake_etcd.cc \
	cpp/util/init.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/masterelection.cc \
	cpp/util/openssl_util.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/protobuf_util.h \
	cpp/util/read_key.cc \
	cpp/util/status.cc \
	cpp/util/sync_task.cc \
	cpp/util/task.cc \
	cpp/util/thread_pool.cc \
	cpp/util/thread_pool.h \
	cpp/util/util.cc \
	cpp/util/uuid.cc \
	cpp/version.cc \
	proto/ct.pb.cc \
	proto/ct.pb.h

cpp_libtest_a_CPPFLAGS = \
	-I$(GMOCK_DIR) \
	-I$(GTEST_DIR) \
	$(AM_CPPFLAGS)
cpp_libtest_a_SOURCES = \
	cpp/gmock-all.cc \
	cpp/gtest-all.cc \
	cpp/util/testing.cc

cpp_server_ct_mirror_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_server_ct_mirror_SOURCES = \
	cpp/client/async_log_client.cc \
	cpp/server/ct-mirror.cc \
	cpp/server/certificate_handler.cc \
	cpp/server/handler.cc \
	cpp/server/json_output.cc \
	cpp/server/server_helper.cc

cpp_server_ct_mirror_v2_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_server_ct_mirror_v2_SOURCES = \
	cpp/client/async_log_client.cc \
	cpp/server/ct-mirror_v2.cc \
	cpp/server/certificate_handler_v2.cc \
	cpp/server/handler_v2.cc \
	cpp/server/json_output.cc \
	cpp/server/server_helper.cc

cpp_server_ct_server_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_server_ct_server_SOURCES = \
	cpp/client/async_log_client.cc \
	cpp/server/ct-server.cc \
	cpp/server/certificate_handler.cc \
	cpp/server/handler.cc \
	cpp/server/json_output.cc \
	cpp/server/log_processes.cc \
	cpp/server/server_helper.cc

cpp_server_ct_server_v2_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_server_ct_server_v2_SOURCES = \
	cpp/client/async_log_client.cc \
	cpp/server/ct-server_v2.cc \
	cpp/server/certificate_handler_v2.cc \
	cpp/server/handler_v2.cc \
	cpp/server/json_output.cc \
	cpp/server/log_processes.cc \
	cpp/server/server_helper.cc

cpp_server_xjson_server_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(icu_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_server_xjson_server_SOURCES = \
	cpp/client/async_log_client.cc \
	cpp/server/handler.cc \
	cpp/server/json_output.cc \
	cpp/server/log_processes.cc \
	cpp/server/server_helper.cc \
	cpp/server/xjson-server.cc \
	cpp/server/x_json_handler.cc \
	cpp/util/json_wrapper.cc \
	third_party/objecthash/objecthash.cc

cpp_tools_ct_clustertool_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf -lsqlite3
cpp_tools_ct_clustertool_SOURCES = \
	cpp/proto/serializer.cc \
	cpp/tools/clustertool_main.cc \
	cpp/util/init.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/read_key.cc \
	cpp/util/util.cc \
	cpp/version.cc

cpp_tools_db_tool_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_tools_db_tool_SOURCES = \
	cpp/proto/serializer.cc \
	cpp/tools/db_tool.cc \
	cpp/util/init.cc \
	cpp/util/util.cc \
	cpp/version.cc

cpp_client_ct_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf -lsqlite3
cpp_client_ct_SOURCES = \
	cpp/client/async_log_client.cc \
	cpp/client/client.cc \
	cpp/client/ct.cc \
	cpp/client/http_log_client.cc \
	cpp/client/ssl_client.cc \
	cpp/monitor/database.cc \
	cpp/monitor/monitor.cc \
	cpp/monitor/sqlite_db.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/init.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/read_key.cc \
	cpp/util/util.cc \
	cpp/version.cc

cpp_server_ct_dns_server_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	${libevent_LIBS} \
	-lprotobuf -lldns -lsqlite3
cpp_server_ct_dns_server_SOURCES = \
	cpp/proto/serializer.cc \
	cpp/server/ct-dns-server.cc \
	cpp/server/event.cc \
	cpp/util/init.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc \
	cpp/version.cc

cpp_tools_dump_cert_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
  ${libevent_LIBS} \
	-lprotobuf
cpp_tools_dump_cert_SOURCES = \
	cpp/tools/dump_cert.cc \
	cpp/util/init.cc \
	cpp/util/util.cc \
	cpp/version.cc

cpp_tools_dump_sth_LDADD = \
	cpp/libcore.a \
	-lprotobuf
cpp_tools_dump_sth_SOURCES = \
	cpp/tools/dump_sth.cc \
	cpp/version.cc

cpp_tools_etcd_watch_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_tools_etcd_watch_SOURCES = \
	cpp/tools/etcd_watch.cc \
	cpp/util/init.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/version.cc

cpp_util_bench_etcd_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS)
cpp_util_bench_etcd_SOURCES = \
	cpp/util/bench_etcd.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc

cpp_util_etcd_masterelection_LDADD = \
	cpp/libcore.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_util_etcd_masterelection_SOURCES = \
	cpp/util/etcd_masterelection.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc

cpp_base_notification_test_LDADD = \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_base_notification_test_SOURCES = \
	cpp/base/notification.cc \
	cpp/base/notification_test.cc

cpp_fetcher_remote_peer_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_fetcher_remote_peer_test_SOURCES = \
	cpp/client/async_log_client.cc \
	cpp/fetcher/remote_peer.cc \
	cpp/fetcher/remote_peer_test.cc \
	cpp/log/test_signer.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_log_cluster_state_controller_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_log_cluster_state_controller_test_SOURCES = \
	cpp/client/async_log_client.cc \
	cpp/log/cluster_state_controller_test.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_log_database_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_log_database_test_SOURCES = \
	cpp/log/database_test.cc \
	cpp/log/test_signer.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_log_etcd_consistent_store_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_log_etcd_consistent_store_test_SOURCES = \
	cpp/log/etcd_consistent_store_test.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_log_file_storage_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_log_file_storage_test_SOURCES = \
	cpp/log/file_storage.cc \
	cpp/log/file_storage_test.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_log_frontend_signer_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_log_frontend_signer_test_SOURCES = \
	cpp/log/frontend_signer_test.cc \
	cpp/log/test_signer.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_log_log_lookup_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_log_log_lookup_test_SOURCES = \
	cpp/log/log_lookup_test.cc \
	cpp/log/test_signer.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_log_log_signer_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_log_log_signer_test_SOURCES = \
	cpp/log/log_signer_test.cc \
	cpp/log/test_signer.cc \
	cpp/proto/serializer.cc \
	cpp/util/util.cc

cpp_log_logged_entry_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_log_logged_entry_test_SOURCES = \
	cpp/log/logged_entry_test.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/util.cc

cpp_log_strict_consistent_store_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_log_strict_consistent_store_test_SOURCES = \
	cpp/log/strict_consistent_store_cert.cc \
	cpp/log/strict_consistent_store_test.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc

cpp_log_tree_signer_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_log_tree_signer_test_SOURCES = \
	cpp/log/test_signer.cc \
	cpp/log/tree_signer_test.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_log_signer_verifier_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_log_signer_verifier_test_SOURCES = \
	cpp/log/signer_verifier_test.cc \
	cpp/log/test_signer.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/util.cc

cpp_monitor_database_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf -lsqlite3
cpp_monitor_database_test_SOURCES = \
	cpp/log/test_signer.cc \
	cpp/monitor/database.cc \
	cpp/monitor/database_test.cc \
	cpp/monitor/sqlite_db.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/util.cc

cpp_monitoring_gcm_exporter_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf -lcrypto -lssl
cpp_monitoring_gcm_exporter_test_SOURCES = \
	cpp/monitoring/gcm/exporter.cc \
	cpp/monitoring/gcm/exporter_test.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/protobuf_util.cc

cpp_monitoring_counter_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_monitoring_counter_test_SOURCES = \
	cpp/monitoring/counter_test.cc \
	cpp/util/protobuf_util.cc

cpp_monitoring_gauge_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_monitoring_gauge_test_SOURCES = \
	cpp/monitoring/gauge_test.cc \
	cpp/util/protobuf_util.cc

cpp_monitoring_registry_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_monitoring_registry_test_SOURCES = \
	cpp/monitoring/registry_test.cc \
	cpp/util/protobuf_util.cc

cpp_net_url_fetcher_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_net_url_fetcher_test_SOURCES = \
	cpp/net/url_fetcher_test.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/protobuf_util.cc
EXTRA_cpp_net_url_fetcher_test_DEPENDENCIES = \
	test/testdata/urlfetcher_test_certs/localhost-key.pem

cpp_proto_serializer_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_proto_serializer_test_SOURCES = \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/proto/serializer_test.cc \
	cpp/util/util.cc

cpp_server_proxy_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_server_proxy_test_SOURCES = \
	cpp/server/json_output.cc \
	cpp/server/proxy.cc \
	cpp/server/proxy_test.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/protobuf_util.cc

cpp_util_bignum_test_LDADD = \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_util_bignum_test_SOURCES = \
	cpp/util/bignum.cc \
	cpp/util/bignum_test.cc

cpp_util_etcd_delete_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS)
cpp_util_etcd_delete_test_SOURCES = \
	cpp/util/etcd_delete_test.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc

cpp_util_etcd_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS)
cpp_util_etcd_test_SOURCES = \
	cpp/util/etcd_test.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc

cpp_util_fake_etcd_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS)
cpp_util_fake_etcd_test_SOURCES = \
	cpp/util/fake_etcd_test.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc
EXTRA_cpp_util_fake_etcd_test_DEPENDENCIES = \
	test/testdata/urlfetcher_test_certs/localhost-key.pem

cpp_util_json_wrapper_test_LDADD = \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS)
cpp_util_json_wrapper_test_SOURCES = \
	cpp/util/json_wrapper.cc \
	cpp/util/json_wrapper_test.cc \
	cpp/util/util.cc

cpp_util_libevent_wrapper_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_util_libevent_wrapper_test_SOURCES = \
	cpp/util/libevent_wrapper.cc \
	cpp/util/libevent_wrapper_test.cc

cpp_util_masterelection_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf
cpp_util_masterelection_test_SOURCES = \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/masterelection_test.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc
EXTRA_cpp_util_masterelection_test_DEPENDENCIES = \
	test/testdata/urlfetcher_test_certs/localhost-key.pem

cpp_merkletree_merkle_tree_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_merkletree_merkle_tree_test_SOURCES = \
	cpp/util/util.cc \
	cpp/merkletree/merkle_tree_test.cc

cpp_merkletree_serial_hasher_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_merkletree_serial_hasher_test_SOURCES = \
	cpp/util/util.cc \
	cpp/merkletree/serial_hasher_test.cc

cpp_merkletree_sparse_merkle_tree_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_merkletree_sparse_merkle_tree_test_SOURCES = \
	cpp/util/util.cc \
	cpp/merkletree/sparse_merkle_tree_test.cc

cpp_merkletree_tree_hasher_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_merkletree_tree_hasher_test_SOURCES = \
	cpp/util/util.cc \
	cpp/merkletree/tree_hasher_test.cc

cpp_merkletree_verifiable_map_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_merkletree_verifiable_map_test_SOURCES = \
	cpp/util/util.cc \
	cpp/merkletree/verifiable_map_test.cc

cpp_util_sync_task_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_util_sync_task_test_SOURCES = \
	cpp/util/sync_task_test.cc

cpp_util_task_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_util_task_test_SOURCES = \
	cpp/util/task_test.cc

cpp_util_thread_pool_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_util_thread_pool_test_SOURCES = \
	cpp/util/thread_pool_test.cc

cpp_log_cert_checker_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_log_cert_checker_test_SOURCES = \
	cpp/log/cert_checker_test.cc \
	cpp/util/util.cc

cpp_log_cert_submission_handler_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	-lprotobuf
cpp_log_cert_submission_handler_test_SOURCES = \
	cpp/log/cert_submission_handler_test.cc \
	cpp/util/util.cc

cpp_log_cert_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_log_cert_test_SOURCES = \
	cpp/log/cert_test.cc \
	cpp/util/util.cc

cpp_log_cms_verifier_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_log_cms_verifier_test_SOURCES = \
	cpp/log/cms_verifier_test.cc \
	cpp/util/util.cc

cpp_log_ct_extensions_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_log_ct_extensions_test_SOURCES = \
	cpp/log/ct_extensions_test.cc \
	cpp/util/util.cc

cpp_log_database_large_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_log_database_large_test_SOURCES = \
	cpp/log/database_large_test.cc \
	cpp/log/test_signer.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_log_frontend_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(json_c_LIBS) \
	$(libevent_LIBS) \
	$(leveldb_LIBS) \
	-lprotobuf -lsqlite3
cpp_log_frontend_test_SOURCES = \
	cpp/log/frontend_test.cc \
	cpp/log/test_signer.cc \
	cpp/proto/cert_serializer.cc \
	cpp/proto/serializer.cc \
	cpp/util/json_wrapper.cc \
	cpp/util/libevent_wrapper.cc \
	cpp/util/periodic_closure.cc \
	cpp/util/protobuf_util.cc \
	cpp/util/util.cc

cpp_merkletree_merkle_tree_large_test_LDADD = \
	cpp/libcore.a \
	cpp/libtest.a \
	$(evhtp_LIBS) \
	$(libevent_LIBS)
cpp_merkletree_merkle_tree_large_test_SOURCES = \
	cpp/merkletree/merkle_tree_large_test.cc \
	cpp/util/util.cc

docker: all
	sudo docker build -t gcr.io/${PROJECT}/super_duper:test .
	sudo docker build -f Dockerfile-ct-mirror -t gcr.io/${PROJECT}/super_mirror:test .
	sudo docker build -t gcr.io/${PROJECT}/etcd:test cloud/etcd
	sudo docker build -t gcr.io/${PROJECT}/prometheus:test cloud/prometheus
