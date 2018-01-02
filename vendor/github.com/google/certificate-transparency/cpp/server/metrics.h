#ifndef CERT_TRANS_SERVER_METRICS_H_
#define CERT_TRANS_SERVER_METRICS_H_

struct evhttp_request;

namespace cert_trans {


void ExportPrometheusMetrics(evhttp_request* req);


}  // namespace cert_trans

#endif  // CERT_TRANS_SERVER_METRICS_H_
