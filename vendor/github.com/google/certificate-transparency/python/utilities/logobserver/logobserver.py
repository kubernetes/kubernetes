#!/usr/bin/env python
import gflags
from google.protobuf import text_format
import logging
import os
import sys
import requests

from ct.cert_analysis import tld_list
from ct.client.db import sqlite_connection as sqlitecon
from ct.client import prober
from ct.client.db import sqlite_log_db
from ct.client.db import sqlite_temp_db
from ct.client.db import sqlite_cert_db
from ct.proto import client_pb2

FLAGS = gflags.FLAGS
gflags.DEFINE_string("ctlog_config", "ct/config/logs.config",
                     "Configuration file for log servers to monitor")
gflags.DEFINE_string("log_level", "WARNING", "logging level")
gflags.DEFINE_string("ct_sqlite_db", "/tmp/ct", "Location of the CT database")
gflags.DEFINE_string("ct_sqlite_temp_dir", "/tmp/ct_tmp", "Directory for "
                     "temporary CT data.")
gflags.DEFINE_string("ct_sqlite_cert_db", "/tmp/ct_cert", "Location of "
                     "certificate database.")
gflags.DEFINE_string("monitor_state_dir", "/tmp/ct_monitor",
                     "Filename prefix for monitor state. State for a given log "
                     "will be stored in a monitor_state_dir/log_id file")

def create_directory(directory):
    if not os.path.exists(directory):
        logging.info("Creating directory: %s" % directory)
        os.makedirs(directory)

if __name__ == '__main__':
    sys.argv = FLAGS(sys.argv)
    logging.basicConfig(level=FLAGS.log_level)

    create_directory(FLAGS.ct_sqlite_temp_dir)
    create_directory(FLAGS.monitor_state_dir)

    try:
        list_ = requests.get(tld_list.TLD_LIST_ADDR, timeout=5)
        if list_.status_code == 200:
            create_directory(FLAGS.tld_list_dir)
            with open('/'.join((FLAGS.tld_list_dir, "tld_list")), 'w') as f:
                f.write(list_.content)
    except requests.exceptions.RequestException:
        logging.warning("Couldn't fetch top level domain list")

    sqlite_log_db = sqlite_log_db.SQLiteLogDB(
        sqlitecon.SQLiteConnectionManager(FLAGS.ct_sqlite_db))

    sqlite_temp_db_factory = sqlite_temp_db.SQLiteTempDBFactory(
        sqlitecon.SQLiteConnectionManager(FLAGS.ct_sqlite_temp_dir + "/meta"),
                                          FLAGS.ct_sqlite_temp_dir)

    sqlite_cert_db = sqlite_cert_db.SQLiteCertDB(
            sqlitecon.SQLiteConnectionManager(FLAGS.ct_sqlite_cert_db))

    ctlogs = client_pb2.CtLogs()
    with open(FLAGS.ctlog_config, "r") as config:
        log_config = config.read()
    text_format.Merge(log_config, ctlogs)

    ct_server_list = []
    for log in ctlogs.ctlog:
        sqlite_log_db.update_log(log)
        ct_server_list.append(log.log_server)

    prober_thread = prober.ProberThread(ctlogs, sqlite_log_db,
                                        sqlite_cert_db,
                                        sqlite_temp_db_factory,
                                        FLAGS.monitor_state_dir)
    prober_thread.start()
