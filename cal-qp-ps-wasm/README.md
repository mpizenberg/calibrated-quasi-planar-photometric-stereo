```sh
wasm-pack build --target web -- --features console_error_panic_hook
```

This will generate a `pkg/` directory with two important files inside:

- `pkg/lowrr_wasm.js`: the "glue" JavaScript module to be imported.
- `pkg/lowrr_wasm_bg.wasm`: the compiled WebAssembly module corresponding to the rust code in `src/lib.rs`.
