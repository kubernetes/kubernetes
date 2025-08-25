import {
    WASI, File, OpenFile, PreopenDirectory, ConsoleStdout
} from "https://esm.sh/@bjorn3/browser_wasi_shim";
const enc = new TextEncoder();

// Arborescence: /home/web/.kube/config
const fsRoot = new PreopenDirectory("/", [
    ["kubeconfig", new File(enc.encode(kubeconfigText))]
]);

const fds = [
    new OpenFile(new File(new Uint8Array())), // stdin
    ConsoleStdout.lineBuffered(msg => console.log(msg)),  // stdout
    ConsoleStdout.lineBuffered(msg => console.warn(msg)), // stderr
    fsRoot                                           // pré-ouverture du FS
];

// args/env comme tu veux, et HOME pour le chemin
const wasi = new WASI(["kube-wasi.wasm", "get", "nodes", "-v=9"], ["KUBECONFIG=/kubeconfig", "KUBERC=off"], fds);

// charge et lance le wasm
const wasm = await WebAssembly.compileStreaming(fetch("kubectl.wasm"));
const instance = await WebAssembly.instantiate(wasm, {
    "wasi_snapshot_preview1": wasi.wasiImport,
});
wasi.start(instance);