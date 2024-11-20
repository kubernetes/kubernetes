const importObject = {};
let wasmInstance = null;


// Try streaming instantiation
WebAssembly.instantiateStreaming(fetch('kubectl'), importObject)
    .then(({ instance }) => {
        wasmInstance = instance;
    })
    .catch(() => {
        // Fallback
        return fetch('kubectl.wasm')
            .then(res => res.arrayBuffer())
            .then(buffer => WebAssembly.instantiate(buffer, importObject))
            .then(({ instance }) => {
                wasmInstance = instance;
            });
    });
