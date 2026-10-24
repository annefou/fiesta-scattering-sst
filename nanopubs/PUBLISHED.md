# Published nanopub chain — URI registry

This file is the canonical registry of published nanopub URIs for this replication.
The chain was published before this ledger existed; the URIs below are copied from the
chain table in `README.md`.

## Chain

| Step | Template | URI | Published |
|---|---|---|---|
| 01 | Quote-with-comment | https://w3id.org/sciencelive/np/RAUqGWdJOve1i0KCOCZhPfXT9qXquiS3qFdLNLgKu-j-I | before this ledger |
| 02 | AIDA Sentence | https://w3id.org/np/RAsnOdj5BxkhD_u67TguhaVIz6T1TaIGW5qzM7_QuNrqU | before this ledger |
| 03 | FORRT Claim | https://w3id.org/sciencelive/np/RAQPvE7Y4PNeL2oDwFh_uJgJbFHyBmWEvyZfO-RWy1pP8 | before this ledger |
| 04 | FORRT Reproduction/Replication Study | https://w3id.org/sciencelive/np/RA45t1bdfz6Jr40G9dDqrWAF4i7-DT3HQNOSheRWjcuho | before this ledger |
| 05 | FORRT Replication Outcome | https://w3id.org/sciencelive/np/RAK2ynlqgA3L_YVbEo-cVoXZ7T63q4eCFZqf9CPWWNc40 | before this ledger |
| 06 | CiTO Citation | https://w3id.org/sciencelive/np/RAA6_hyyQvv4h0l4bFNt-9tEn2KRRIL4mVE0pRzt4W31k | before this ledger |

Step 02 is on `https://w3id.org/np/…` rather than `…/sciencelive/np/…`: it was published via
Nanodash because the Science Live AIDA form failed when both *datasets* and *publications*
were filled (see README). Both namespaces are valid and citable.

## Verification

`verify_chain` (forrt-research MCP) run on 2026-10-08 against both the production and the
dev Science Live API: all steps reachable, the Outcome's archived version DOI resolves, and the
cited DOI resolves — green.
