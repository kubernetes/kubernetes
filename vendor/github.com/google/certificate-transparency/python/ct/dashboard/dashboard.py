#!/usr/bin/env python
import gflags
from google.protobuf import text_format
import logging
import os
import sys
import time
from wsgiref import simple_server

from ct.dashboard import grapher
from ct.client.db import sqlite_connection as sqlitecon
from ct.client import prober
from ct.client.db import sqlite_cert_db
from ct.client.db import sqlite_log_db
from ct.client.db import sqlite_temp_db
from ct.proto import client_pb2


FLAGS = gflags.FLAGS

gflags.DEFINE_string("dashboard_host", "127.0.0.1", "Dashboard server host")
gflags.DEFINE_integer("dashboard_port", 8000, "Dashboard server port")
gflags.DEFINE_string("ctlog_config", "ct/config/logs.config",
                     "Configuration file for log servers to monitor")
# TODO(ekasper): clean up storage configuration so there's one base directory.
gflags.DEFINE_string("ct_sqlite_db", "/tmp/ct", "Location of the CT database")
gflags.DEFINE_string("ct_sqlite_cert_db", "/tmp/ct_cert", "Location of "
                     "certificate database.")
gflags.DEFINE_string("ct_sqlite_temp_dir", "/tmp/ct_tmp", "Directory for "
                     "temporary CT data.")
gflags.DEFINE_string("monitor_state_dir", "/tmp/ct_monitor",
                     "Filename prefix for monitor state. State for a given log "
                     "will be stored in a monitor_state_dir/log_id file")
gflags.DEFINE_string("log_level", "WARNING", "logging level")

class DashboardServer(object):
    template_pages = {"default", "sth"}

    """A simple web application for serving status pages."""
    def __init__(self, host, port, db, ct_server_list):
        self.host = host
        self.port = port
        self.db = db
        self.grapher = grapher.GvizGrapher()
        self.templates = {}
        self._read_templates()
        self.favicon = ""
        self._read_favicon()

    def _read_templates(self):
        """Read the HTML templates from the templates/ directory."""
        for page in DashboardServer.template_pages:
            template_file = os.path.join(os.path.dirname(__file__),
                                         "templates/%s.html" % page)
            self.templates[page] = open(template_file, "r").read()

    def _read_favicon(self):
        """Read favicon.ico."""
        favicon_file = os.path.join(os.path.dirname(__file__), "favicon.ico")
        self.favicon = open(favicon_file, "rb").read()

    def __repr__(self):
        return "%r(%r:%r)" % (self.__class__.__name__,
                              self.host, self.port)

    def __str__(self):
        return "%s(%s:%d)" % (self.__class__.__name__,
                              self.host, self.port)

    def __call__(self, environ, start_response):
        page = environ.get("PATH_INFO", "").strip("/")
        # Serve the default page when there is no match
        status = "200 OK"
        if page == "favicon.ico":
                headers = [("Content-type", "image/png")]
                start_response(status, headers)
                return self.favicon

        status = "200 OK"
        headers = [("Content-type", "text/html")]
        start_response(status, headers)
        varz = {}
        varz["current_time"] = "\nCurrent time: %s\n" % time.strftime("%c")
        if page == "sth":
            self._fill_sth_template(varz)
            return self.templates["sth"] % varz
        else:
            self._fill_default_template(varz)
            return self.templates["default"] % varz

    def _fill_default_template(self, varz):
        """Update the varz with variables required by the default template."""
        varz["pages"] = """<a href=http://%s:%d/sth>
                        The Signed Tree Head Dashboard</a>\n""" % (
            self.host, self.port)

    def _fill_sth_template(self, varz):
        # TODO(ekasper): graph all logs
        server_id = ct_server_list[0]
        varz["log"] = server_id.encode("ascii")
        one_week_ago = (time.time() - 7*24*60*60)*1000
        columns = [("sth.timestamp", "timestamp_ms", "Timestamp"),
                   ("sth.tree_size", "number", "Tree size")]
        sth_to_show = self.db.scan_latest_sth_range(server_id,
                                                    start=one_week_ago,
                                                    limit=1000)
        varz["json"] = self.grapher.make_table(sth_to_show, columns,
                                               order_by="timestamp")

    def run(self):
        """Set up the HTTP server and loop forever."""
        httpd = simple_server.make_server(self.host, self.port, self)
        logging.info("Serving on http://%s:%d" % (self.host, self.port))
        httpd.serve_forever()

if __name__ == "__main__":
    sys.argv = FLAGS(sys.argv)
    logging.basicConfig(level=FLAGS.log_level)

    sqlite_log_db = sqlite_log_db.SQLiteLogDB(
        sqlitecon.SQLiteConnectionManager(FLAGS.ct_sqlite_db))

    sqlite_cert_db = sqlite_cert_db.SQLiteCertDB(
        sqlitecon.SQLiteConnectionManager(FLAGS.ct_sqlite_cert_db))

    if not os.path.exists(FLAGS.ct_sqlite_temp_dir):
        logging.info("Creating directory for temporary data at %s" %
                     FLAGS.ct_sqlite_temp_dir)
        os.makedirs(FLAGS.ct_sqlite_temp_dir)

    if not os.path.exists(FLAGS.monitor_state_dir):
        logging.info("Creating directory for monitor state at %s" %
                     FLAGS.monitor_state_dir)
        os.makedirs(FLAGS.monitor_state_dir)

    sqlite_temp_db_factory = sqlite_temp_db.SQLiteTempDBFactory(
        sqlitecon.SQLiteConnectionManager(FLAGS.ct_sqlite_temp_dir + "/meta"),
                                          FLAGS.ct_sqlite_temp_dir)

    ctlogs = client_pb2.CtLogs()
    with open(FLAGS.ctlog_config, "r") as config:
        log_config = config.read()
    text_format.Merge(log_config, ctlogs)

    ct_server_list = []
    for log in ctlogs.ctlog:
        sqlite_log_db.update_log(log)
        ct_server_list.append(log.log_server)

    prober_thread = prober.ProberThread(ctlogs,
                                        sqlite_log_db,
                                        sqlite_cert_db,
                                        sqlite_temp_db_factory,
                                        FLAGS.monitor_state_dir)
    prober_thread.daemon = True
    prober_thread.start()
    server = DashboardServer(FLAGS.dashboard_host, FLAGS.dashboard_port,
                             sqlite_log_db, ct_server_list)
    server.run()
