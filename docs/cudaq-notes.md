# CUDA-Q Implementation Notes

## Kernel Language Constraints
- Mark every executable quantum routine with `@cudaq.kernel`. Regular Python helper functions cannot be invoked inside kernels; re-express shared logic as kernels that take `cudaq.qview` arguments or inline it.
- Kernels allow a restricted Python subset: no dynamic list creation (`[]`), appends, or `None` placeholders. Allocate fixed-size `cudaq.qvector` or `cudaq.qarray` objects and branch on integer literals known at compile time.
- Classical control must be static; loops must have determinable bounds (e.g., `for i in range(4)`). Input parameters are passed by value, so host data should be preprocessed before kernel invocation.

## Measurement Ordering
- Qubits are measured in allocation order. When allocating multiple registers (e.g., `data = cudaq.qvector(4)` followed by `anc = cudaq.qvector(2*k)`), the resulting bitstrings place the `data` bits first, then ancilla bits. Post-selection logic must slice accordingly.

## Noise Modeling
- Backend-level noise works via `cudaq.NoiseModel` plus `add_all_qubit_channel("gate", channel, num_controls=0/1, ...)`. Use depolarizing channels to emulate Pauli errors described in arXiv:2601.07223 and attach them to every gate that should experience faults.
- Alternatively, location-dependent faults can be injected within kernels through `cudaq.apply_noise(qubit, channel)` when modeling specific error events (readout, helper ancillas, etc.).

## Recommended Targets
- `density-matrix-cpu` or `dm-sim` targets faithfully incorporate the user-defined noise models; state-vector simulators ignore them. Use `cudaq.set_target("density-matrix-cpu")` before training.

## Workflow Tips
1. Build kernels with pure quantum instructions only; keep dataset prep, logical decoding, and optimizer logic on the host side.
2. When logical encoding requires repeated patterns (e.g., [[4,2,2]] code preparation), define separate kernels or inline the operations within the main kernel to satisfy the compiler.
3. Guard encoded-mode ancilla usage by building distinct kernels for each syndrome-round count to avoid dynamic register allocation within device code.
