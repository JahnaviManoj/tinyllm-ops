"""SMS template shapes for the synthetic data generator (tutorial 1.2).

Each Template is the *skeleton* of a real bank SMS with every concrete value
replaced by a named placeholder from a fixed vocabulary:

    {amount} {bal}   -> money values
    {tail}           -> account/card last digits
    {merchant} {vpa} {name} -> payee/payer
    {bank} {date} {ref} {otp} {phone} {url} -> the rest

Sources (see docs/decisions.md for the in-distribution vs OOD split):
  - akhilnarang/bank-sms-parser fixtures (sanitized, 11 Indian banks)
  - YuvrajDube/Bank-SMS-Parser samples.json (fake values)
  - hand-written: scam/phishing, Hinglish, ATM and wallet gap-fillers

The Kaggle dataset was reviewed but not used: its messages are UAE-bank
(AED) shapes and this project targets Indian bank SMS only.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    shape: str
    channel: str | None  # a Channel enum value, or None for negatives
    hint_txn_type: str | None = None  # "debit" / "credit" / None
    is_negative: bool = False  # OTP/promo/reminder -> is_transaction=false
    is_scam: bool = False  # phishing shape -> is_suspected_scam=true


TEMPLATES = [
    # ------------------------------------------------------------------ UPI
    # HDFC multi-line UPI debit
    Template(
        "Sent Rs.{amount}\nFrom {bank} A/C *{tail}\nTo {merchant}\nOn {date}\n"
        "Ref {ref}\nNot You?\nCall {phone}/SMS BLOCK UPI to {phone}",
        channel="UPI",
        hint_txn_type="debit",
    ),
    # ICICI single-line "; X credited" style
    Template(
        "{bank} Acct XX{tail} debited for Rs {amount} on {date}; {merchant} credited. "
        "UPI:{ref}. Call {phone} for dispute. SMS BLOCK {tail} to {phone}.",
        channel="UPI",
        hint_txn_type="debit",
    ),
    # IDFC "debited by ...; NAME credited. RRN" with balance tail
    Template(
        "Your A/c XX{tail} debited by Rs. {amount} on {date}; {name} credited. "
        "RRN {ref}. Available balance Rs. {bal}. Team {bank}",
        channel="UPI",
        hint_txn_type="debit",
    ),
    # IndusInd VPA-to-VPA
    Template(
        "A/C *XX{tail} debited by Rs {amount} towards {vpa}. RRN:{ref}. "
        "Avl Bal:{bal}. Not you? Call {phone} - {bank}",
        channel="UPI",
        hint_txn_type="debit",
    ),
    # slice fintech phrasing
    Template(
        "Rs. {amount} sent from a/c xx{tail} on {date} to {name} (UPI Ref: {ref}). "
        "Not you? Call {phone} - {bank}",
        channel="UPI",
        hint_txn_type="debit",
    ),
    # UPI info-string style (UPI/P2A/...)
    Template(
        "Rs {amount} debited from A/c XX{tail} via UPI on {date}. "
        "UPI/P2A/{vpa}/Personal. UPI Ref: {ref}.",
        channel="UPI",
        hint_txn_type="debit",
    ),
    # HDFC UPI credit alert
    Template(
        "Credit Alert!\nRs.{amount} credited to {bank} A/c XX{tail} on {date} "
        "from VPA {vpa} (UPI {ref})",
        channel="UPI",
        hint_txn_type="credit",
    ),
    # ICICI UPI credit
    Template(
        "Dear Customer, Acct XX{tail} is credited with Rs {amount} on {date} "
        "from {name}. UPI:{ref}-{bank}.",
        channel="UPI",
        hint_txn_type="credit",
    ),
    # slice UPI credit with balance tail
    Template(
        "Rs. {amount} received in {bank} A/c xx{tail} on {date} from {name} via UPI "
        "(Ref ID: {ref}). Avl. Bal. Rs. {bal} - {bank}",
        channel="UPI",
        hint_txn_type="credit",
    ),
    # ----------------------------------------------------------------- card
    # HDFC compressed cc_spend
    Template(
        "Spent Rs.{amount} On {bank} Card {tail} At {merchant} On {date}.Not You? "
        "To Block+Reissue Call {phone}/SMS BLOCK CC {tail} to {phone}",
        channel="card",
        hint_txn_type="debit",
    ),
    # HDFC debit-card spend with balance tail
    Template(
        "Spent Rs.{amount} From {bank} Card x{tail} At {merchant} On {date} "
        "Bal Rs.{bal} Not You? Call {phone}/SMS BLOCK DC {tail} to {phone}",
        channel="card",
        hint_txn_type="debit",
    ),
    # Axis multi-line spend
    Template(
        "Spent INR {amount}\n{bank} Card no. XX{tail}\n{date}\n{merchant}\n"
        "Avl Limit: INR {bal}\nNot you? SMS BLOCK {tail} to {phone}",
        channel="card",
        hint_txn_type="debit",
    ),
    # SBI "Trxn. not done by you?" tail
    Template(
        "Rs.{amount} spent on your {bank} Credit Card ending {tail} at {merchant} "
        "on {date}. Trxn. not done by you? Report at {url}",
        channel="card",
        hint_txn_type="debit",
    ),
    # ICICI cc spend with limit
    Template(
        "INR {amount} spent using {bank} Card XX{tail} on {date} on {merchant}. "
        "Avl Limit: INR {bal}. If not you, call {phone}/SMS BLOCK {tail} to {phone}.",
        channel="card",
        hint_txn_type="debit",
    ),
    # Equitas "Available limit is INR." quirk
    Template(
        "INR {amount} spent on {bank} CC XX{tail} at {merchant} on {date}. "
        "Available limit is INR. {bal}. Not you? SMS BLOCK CC {tail} to {phone}",
        channel="card",
        hint_txn_type="debit",
    ),
    # Jupiter casual fintech phrasing
    Template(
        "Hey there, you've spent ₹{amount} to {merchant} on your {bank} Credit Card "
        "ending {tail} on {date}. Tap to view your transactions in the {bank} app.",
        channel="card",
        hint_txn_type="debit",
    ),
    # OneCard playful spend
    Template(
        "You've spent Rs. {amount} at {merchant} with your {bank} Credit Card ending "
        "in XX{tail}. Reward points are now in your basket. To dispute this payment, "
        "click: {url}",
        channel="card",
        hint_txn_type="debit",
    ),
    # foreign-currency spend
    Template(
        "USD {amount} spent on your {bank} Card XX{tail} at {merchant} on {date}. "
        "Foreign currency markup of 3.5% will be applied. INR equivalent will appear "
        "in statement.",
        channel="card",
        hint_txn_type="debit",
    ),
    # debit-card purchase, no merchant named
    Template(
        "INR {amount} debited from your A/C {tail} towards Debit Card Purchase. "
        "Avl BAL INR {bal} - Not you? Call {phone} to report issue - {bank}",
        channel="card",
        hint_txn_type="debit",
    ),
    # truncated mid-sentence (carrier cut the message off)
    Template(
        "Spent Rs.{amount} on {bank} Card XX{tail} at {merchant} on {date}. Avl L",
        channel="card",
        hint_txn_type="debit",
    ),
    # merchant refund
    Template(
        "Alert! Rs. {amount} refunded by {merchant} on {date} & adjusted against "
        "{bank} Credit Card {tail} View updated balance here: {url}",
        channel="card",
        hint_txn_type="credit",
    ),
    # cc bill payment received
    Template(
        "Payment of INR {amount} has been received towards your {bank} Credit Card "
        "XX{tail} on {date} - {bank}",
        channel="card",
        hint_txn_type="credit",
    ),
    # reversal, no-space-after-bang style
    Template(
        "Transaction Reversed!On {bank} CREDIT Card xx{tail} Amt: Rs.{amount} "
        "By {merchant} On {date}",
        channel="card",
        hint_txn_type="credit",
    ),
    # OneCard playful payment received
    Template(
        "Hola! that was sweet. We have received payment against your {bank} card "
        "for Rs. {amount} on {date}.",
        channel="card",
        hint_txn_type="credit",
    ),
    # ---------------------------------------- netbanking (IMPS/NEFT/RTGS/…)
    # HDFC multi-line IMPS outward
    Template(
        "IMPS INR {amount}\nsent from {bank} A/c XX{tail} on {date}\n"
        "To A/c xxxxxxxxxx{tail}\nRef-{ref}\nNot you?Call {phone}/SMS BLOCK OB to {phone}",
        channel="netbanking",
        hint_txn_type="debit",
    ),
    # IndusInd IMPS debit, "will be credited" future phrasing
    Template(
        "Your account XXXXXXX{tail} debited with Rs. {amount} on {date} and account "
        "XXXXXXX{tail}/{name} will be credited.(IMPS Ref no. {ref}). "
        "Call {phone} to report issue - {bank}",
        channel="netbanking",
        hint_txn_type="debit",
    ),
    # ICICI IMPS credit
    Template(
        "{bank} Account XX{tail} is credited with Rs {amount} on {date} by Account "
        "linked to mobile number XXXXX{tail}. IMPS Ref. no. {ref}.",
        channel="netbanking",
        hint_txn_type="credit",
    ),
    # IDFC IMPS credit with parenthesised ref
    Template(
        "Your a/c no. XXXXXXXX{tail} is credited by Rs. {amount} on {date} by a/c "
        "linked to mobile XXXXXXXXX{tail} (IMPS Ref no {ref} ). Team {bank}",
        channel="netbanking",
        hint_txn_type="credit",
    ),
    # HDFC NEFT deposit with clearing disclaimer
    Template(
        "Update! INR {amount} deposited in {bank} A/c XX{tail} on {date} for NEFT "
        "Cr-{ref}-{name}.Avl bal INR {bal}. Cheque deposits in A/C are subject to clearing",
        channel="netbanking",
        hint_txn_type="credit",
    ),
    # IDFC NEFT debit with Info: line
    Template(
        "Your A/c XXXXXXX{tail} has been debited by Rs. {amount} on {date}. "
        "Info: NEFT/ {ref}/{name}.\nNew bal: Rs. {bal}. Team {bank}",
        channel="netbanking",
        hint_txn_type="debit",
    ),
    # RTGS initiated, no balance
    Template(
        "RTGS txn initiated: Of Rs.{amount} from your {bank} A/c XX{tail} using "
        "Online Banking. Not you?Call {phone}/SMS BLOCK OB to {phone}",
        channel="netbanking",
        hint_txn_type="debit",
    ),
    # generic credited-with-new-balance
    Template(
        "Your A/C XXXXX{tail} is credited with INR {amount} on {date}. "
        "Your new balance is INR {bal}. Team {bank}",
        channel="netbanking",
        hint_txn_type="credit",
    ),
    # salary credit
    Template(
        "Dear Customer, Rs {amount} credited to your A/c XX{tail} on {date} by "
        "SALARY-{name}. Avl Bal: Rs {bal}.",
        channel="netbanking",
        hint_txn_type="credit",
    ),
    # SIP auto-debit
    Template(
        "Your SIP of Rs {amount} in {name} folio {ref} has been debited from "
        "A/c XX{tail} on {date}.",
        channel="netbanking",
        hint_txn_type="debit",
    ),
    # e-mandate redeemed
    Template(
        "Dear Customer, the mandate of INR {amount} raised by {name} on {date} and "
        "is successfully redeemed through RRN {ref} -{bank}.",
        channel="netbanking",
        hint_txn_type="debit",
    ),
    # ------------------------------------------------------------------ ATM
    Template(
        "Rs.{amount} withdrawn at {bank} ATM on {date} from A/c XX{tail}. "
        "Avl Bal Rs.{bal}. Not you? Call {phone}.",
        channel="ATM",
        hint_txn_type="debit",
    ),
    Template(
        "Dear Customer, Rs.{amount} is debited from A/c XX{tail} on {date} for ATM "
        "cash withdrawal at {merchant} ATM. Avl Bal: Rs.{bal} - {bank}",
        channel="ATM",
        hint_txn_type="debit",
    ),
    # cash deposit at ATM
    Template(
        "Cash deposit of Rs. {amount} at {bank} ATM on {date} has been credited to "
        "A/c XX{tail}. Avl Bal: Rs. {bal}.",
        channel="ATM",
        hint_txn_type="credit",
    ),
    # --------------------------------------------------------------- wallet
    Template(
        "Rs.{amount} paid from your {bank} Wallet to {merchant} on {date}. "
        "Order ID {ref}. Wallet balance: Rs.{bal}.",
        channel="wallet",
        hint_txn_type="debit",
    ),
    Template(
        "₹{amount} added to your {bank} Wallet from card XX{tail} on {date}. "
        "Updated wallet balance is ₹{bal}.",
        channel="wallet",
        hint_txn_type="credit",
    ),
    Template(
        "Yay! You've received ₹{amount} cashback in your {bank} wallet for your "
        "order at {merchant}. Ref {ref}.",
        channel="wallet",
        hint_txn_type="credit",
    ),
    # ------------------------------------------------------------- Hinglish
    Template(
        "Aapke A/c XX{tail} se Rs.{amount} debit hua {date} ko {merchant} ke liye. "
        "UPI Ref {ref}. Shak ho toh {phone} par call karein. -{bank}",
        channel="UPI",
        hint_txn_type="debit",
    ),
    Template(
        "Dhanyavaad! Aapke {bank} account XX{tail} mein Rs {amount} jama hue hain "
        "{date} ko. Updated balance: Rs {bal}.",
        channel="netbanking",
        hint_txn_type="credit",
    ),
    # ------------------------------------- negatives (is_transaction=false)
    Template(
        "Use {otp} as your OTP for {bank} Net Banking login. Valid for 5 mins. "
        "Do NOT share with anyone.",
        channel=None,
        is_negative=True,
    ),
    Template(
        "{otp} is the OTP for txn of Rs.{amount} at {merchant} on your {bank} Card "
        "XX{tail}. Valid for one use only. Do not share it with anyone.",
        channel=None,
        is_negative=True,
    ),
    Template(
        "Get flat 50% off + extra 10% cashback on travel bookings with {bank} Credit "
        "Cards this weekend. T&C apply. Visit {url}.",
        channel=None,
        is_negative=True,
    ),
    Template(
        "Your {bank} Credit Card xx{tail} bill of Rs {amount} is due on {date}. "
        "View your bill at {url}.",
        channel=None,
        is_negative=True,
    ),
    Template(
        "Payment of {bank} Credit Card {tail} is due on {date}. Min due Rs {amount} "
        "Total due Rs {bal}. Non payment is reported to Credit Bureau. Pls ignore if paid",
        channel=None,
        is_negative=True,
    ),
    Template(
        "Avl Bal in your A/C XX{tail} as on {date} is INR {bal}. "
        "Call {phone} for details.",
        channel=None,
        is_negative=True,
    ),
    Template(
        "Your {merchant} order with tracking ID {ref} is out for delivery and will "
        "arrive today by {date}. Track: {url}",
        channel=None,
        is_negative=True,
    ),
    Template(
        "Hi, Hope you love using your {bank} Credit Card! Your bill of Rs.{amount} is "
        "ready. Please pay by {date} through the {bank} app. View statement: {url}",
        channel=None,
        is_negative=True,
    ),
    Template(
        "Transaction Declined: Attempt to spend Rs. {amount} on your {bank} Credit "
        "Card XX{tail} at {merchant} was declined due to insufficient credit limit.",
        channel=None,
        is_negative=True,
    ),
    Template(
        "Dear Customer, Rs {amount} will be auto debited via E-Mandate from your "
        "{bank} Card XX{tail} on {date} for {merchant}. Please maintain sufficient limit.",
        channel=None,
        is_negative=True,
    ),
    Template(
        "Dear customer, your attempt to login to {bank} Mobile App failed due to "
        "incorrect MPIN. If not you, call us at {phone} - {bank}",
        channel=None,
        is_negative=True,
    ),
    # -------------------------------- scams (also is_transaction=false)
    Template(
        "Dear customer your {bank} account will be BLOCKED today. Update KYC "
        "immediately at {url} to avoid suspension.",
        channel=None,
        is_negative=True,
        is_scam=True,
    ),
    Template(
        "Congratulations {name}! Your mobile number has won Rs {amount} in the "
        "{bank} Lucky Draw. To claim prize send your account details to {phone} "
        "immediately.",
        channel=None,
        is_negative=True,
        is_scam=True,
    ),
    Template(
        "ALERT: Your {bank} debit card has been suspended due to suspicious "
        "activity. Reactivate now at {url} or call {phone} to avoid permanent block.",
        channel=None,
        is_negative=True,
        is_scam=True,
    ),
    Template(
        "You have received a payment request of Rs.{amount} from {vpa}. Approve in "
        "your UPI app to RECEIVE the money. -{bank}",
        channel=None,
        is_negative=True,
        is_scam=True,
    ),
    Template(
        "Dear user, your {bank} reward points worth Rs.{amount} are expiring today! "
        "Redeem now at {url} to get cash in your account. Hurry!",
        channel=None,
        is_negative=True,
        is_scam=True,
    ),
]
