"""Behavioral / invariance suite, CheckList-style (tutorial 2.2, part 2).

Aggregate metrics say *how much* the model is right; these say *in what ways*
it is wrong. Each behavior is a triple:

    (name, transform: str -> str, assertion: ExpenseRecord -> bool)

`transform` perturbs an SMS, `assertion` states what must hold about the parse
of the perturbed SMS. Nothing executes this list yet — a model to run it
against only arrives in 2.5. Defining it is the 2.2 deliverable; the per-
behavior pass rates land in the eval JSON report later, and the Stage 3
promotion gate checks them so a model can't ship on a good average while one
behavior regressed.

The transforms are string edits, so they only bite on an SMS carrying the
values they look for. BASE_SMS is that anchor: the canonical in-distribution
debit shape every behavior below is written against.
"""

from tinyllm.schema import Category

BASE_SMS = (
    "Sent Rs.450.00\nFrom HDFC Bank A/C *1234\nTo SWIGGY\nOn 12-07-25\n"
    "Ref 519374026611\nNot You?\nCall 18002586161"
)

BEHAVIORS = [
    # perturb the amount → the parsed amount must track it
    (
        "amount_tracks",
        lambda sms: sms.replace("450.00", "999.00"),
        lambda rec: rec.amount == "999.00",
    ),
    # swap the merchant for a same-category one → category must not flip
    (
        "merchant_swap_keeps_category",
        lambda sms: sms.replace("SWIGGY", "ZOMATO"),
        lambda rec: rec.category == Category.food,
    ),
    # append promo noise → still recognized as a transaction
    (
        "promo_noise_ignored",
        lambda sms: sms + " Win Rs.1 crore! Click bit.ly/xyz",
        lambda rec: rec.is_transaction is True,
    ),
    # same amount, different formatting → identical parse
    (
        "amount_format_invariant",
        lambda sms: sms.replace("Rs.450.00", "INR 450.00"),
        lambda rec: rec.amount == "450.00",
    ),
]
