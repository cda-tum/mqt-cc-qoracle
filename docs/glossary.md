# Glossary

Terms that the QOracle extension adds to the
[MQT Core glossary](https://github.com/munich-quantum-toolkit/core/blob/main/docs/glossary.md).
The same rules apply: name the preferred term, list accepted aliases, and keep
each entry readable without knowledge of the implementation.

```{glossary}
:sorted:

QOracle
  MQT's dialect for quantum oracles described by the classical function they
  compute. A `qoracle.oracle` operation applies a reversible function, written
  on integer aliases of QC qubits, and lowering synthesizes its circuit.

oracle alias
  **Preferred term:** oracle alias. **Accepted aliases:** none. The classical
  integer that stands for one `qoracle.oracle` operand inside the oracle's
  region: i1 for a qubit and i<n> for a memref of n qubits, with element 0 as
  the least significant bit.

kept operand
  **Preferred term:** kept operand. **Accepted aliases:** none. A
  `qoracle.oracle` operand that the oracle reads but does not change on
  computational basis states. For superposition inputs, a kept operand can
  become entangled with the consumed operands.

consumed operand
  **Preferred term:** consumed operand. **Accepted aliases:** none. A
  `qoracle.oracle` operand that the oracle overwrites in place with the value
  its region yields. Unlike the use of "consume" under value semantics, the
  operand is a QC reference and stays valid after the operation.

auxiliary qubit
  **Preferred term:** auxiliary qubit. **Accepted aliases:** none. A qubit that
  a lowering introduces to hold intermediate values and returns to |0⟩ before
  the lowered operation ends. The literature also calls it an ancilla; use
  auxiliary qubit in MQT Core prose.
```
