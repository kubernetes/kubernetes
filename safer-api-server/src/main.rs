use http_body_util::{BodyExt, Full};
use hyper::body::{Bytes, Incoming};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use std::net::SocketAddr;
use tokio::net::TcpListener;

const BANNER: &str = r#"
 ____   __   ____  ____  ____     __   ____  __    ____  ____  ____  _  _  ____  ____
/ ___) / _\ (  __)(  __)(  _ \   / _\ (  _ \(  )  / ___)(  __)(  _ \/ )( \(  __)(  _ \
\___ \/    \ ) _)  ) _)  )   /  /    \ ) __/ )(   \___ \ ) _)  )   /\ \/ / ) _)  )   /
(____/\_/\_/(__)  (____)(__\_)  \_/\_/(__)  (__)  (____/(____)(__\_) \__/ (____)(__\_)

  Kubernetes API Server - now mass_memory_safe(tm)
  "We rewrote it in Rust so you don't have to"

  WARNING: This is NOT a real rewrite. We just proxy to the Go one.
           But hey, at least THIS part won't segfault. Happy April Fools!
"#;

const UPSTREAM: &str = "http://localhost:8080";

async fn proxy(req: Request<Incoming>) -> Result<Response<Full<Bytes>>, hyper::Error> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    eprintln!(
        "\x1b[32m[SAFER]\x1b[0m {} {} -> proxying to unsafe Go server (yikes)",
        method, path
    );

    let client = Client::builder(TokioExecutor::new()).build_http();
    let uri = format!("{}{}", UPSTREAM, path);

    let proxy_req: Request<Full<Bytes>> = Request::builder()
        .method(method)
        .uri(&uri)
        .body(Full::new(req.collect().await?.to_bytes()))
        .unwrap();

    match client.request(proxy_req).await {
        Ok(resp) => {
            let status = resp.status();
            let body = resp.into_body().collect().await?.to_bytes();
            eprintln!(
                "\x1b[32m[SAFER]\x1b[0m Response: {} (safely proxied, zero CVEs added)",
                status
            );
            Ok(Response::builder()
                .status(status)
                .header("X-Powered-By", "Rust+Tokio (the safe parts)")
                .header("X-April-Fools", "2026")
                .body(Full::new(body))
                .unwrap())
        }
        Err(_) => {
            let body = r#"{"kind":"Status","apiVersion":"v1","status":"Failure","message":"upstream Go server unreachable - see, this is why we need Rust","reason":"SaferServerError","code":502}"#.to_string();
            Ok(Response::builder()
                .status(502)
                .header("Content-Type", "application/json")
                .header("X-Powered-By", "Rust+Tokio (the safe parts)")
                .body(Full::new(Bytes::from(body)))
                .unwrap())
        }
    }
}

#[tokio::main]
async fn main() {
    eprintln!("{}", BANNER);

    let addr = SocketAddr::from(([0, 0, 0, 0], 6443));
    let listener = TcpListener::bind(addr).await.unwrap();

    eprintln!(
        "\x1b[32m[SAFER]\x1b[0m Listening on {} (memory-safe since boot!)",
        addr
    );
    eprintln!(
        "\x1b[32m[SAFER]\x1b[0m Proxying to {} (the legacy unsafe one)",
        UPSTREAM
    );
    eprintln!("\x1b[32m[SAFER]\x1b[0m 0 panics | 0 segfaults | 0 data races | mass vibes\n");

    loop {
        let (stream, remote) = listener.accept().await.unwrap();
        eprintln!("\x1b[32m[SAFER]\x1b[0m Connection from {} (safely accepted)", remote);

        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service_fn(proxy))
                .await
            {
                eprintln!("\x1b[31m[SAFER]\x1b[0m Error: {} (but at least it was a safe error)", e);
            }
        });
    }
}
