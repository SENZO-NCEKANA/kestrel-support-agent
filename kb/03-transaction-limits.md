---
doc_id: KB-LIM-003
title: Transaction Limits
version: 2.4
owner: Payments Product
effective: 2026-03-01
customer_facing: true
---

# Transaction Limits

## Daily Limits by Verification Level

Every Kestrel account carries a FICA verification level. The level is set when
identity documents are accepted and is independent of the pricing tier the
customer pays for. These are the hard daily caps by level.

| Transaction type | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| ATM withdrawal | R2 000 | R5 000 | R10 000 |
| Card purchase | R5 000 | R20 000 | R50 000 |
| EFT / PayShap out | R3 000 | R25 000 | R100 000 |
| International transfer | Not permitted | R10 000 | R50 000 |

Level 1 is assigned at signup from a verified mobile number alone. Level 2
requires an accepted identity document and proof of address. Level 3 additionally
requires source-of-funds documentation and is granted on request.

## Tier Ceilings

The pricing tier sets a second, separate ceiling. A customer on Private has a
higher ceiling than one on Blue, but a tier ceiling never overrides a
verification cap.

| Limit | Blue | Plus | Private |
|---|---|---|---|
| ATM withdrawal, daily | R3 000 | R6 000 | R10 000 |
| Card purchase, daily | R20 000 | R40 000 | R80 000 |
| EFT / PayShap out, daily | R20 000 | R50 000 | R250 000 |
| International transfer, daily | R10 000 | R25 000 | R100 000 |

## Which Limit Applies

Where the verification cap and the tier ceiling differ, the lower of the two
applies. This is the single most common source of limit complaints: the pricing
page quotes the tier ceiling, which is the number the customer remembers, while
their account may still sit at a lower verification level.

**Worked example.** A Private-tier customer verified only to Level 1 asks why an
R8 000 ATM withdrawal was declined. The Private tier ceiling is R10 000 per day,
but Level 1 caps ATM withdrawals at R2 000. The lower figure governs, so the
verification level is the binding constraint, not the tier. The fix is to
complete Level 2 verification — upgrading the tier changes nothing.

Never quote a tier ceiling without checking the verification level first. A
ceiling the customer cannot actually reach is worse than no answer at all.

## Temporary Limit Increases

A customer may request a temporary increase for a single calendar day. Increases
are capped at the tier ceiling and can never exceed the verification cap.

| Request | Channel | Turnaround |
|---|---|---|
| Within tier ceiling | App self-service | Immediate |
| Above tier ceiling | Agent, reason recorded | 1 business day |
| Above verification cap | Not possible | Verification upgrade required |

Limits reset at midnight SAST. A declined transaction does not consume limit
headroom, but a reversed transaction does not release it until settlement.
