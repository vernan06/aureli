from __future__ import annotations
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from aureli.persistence.db import connect

def to_minor(amount) -> int:
    value = Decimal(str(amount))
    if not value.is_finite() or value <= 0: raise ValueError("Amount must be a positive finite number.")
    return int((value * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

def _category_id(db, user_id: int, category: str, kind: str) -> int:
    name=category.strip()
    if kind not in {"income","expense"} or not name: raise ValueError("Choose a valid category.")
    db.execute("INSERT OR IGNORE INTO categories(user_id,name,kind) VALUES(?,?,?)",(user_id,name,kind))
    return int(db.execute("SELECT id FROM categories WHERE user_id=? AND name=? AND kind=?",(user_id,name,kind)).fetchone()[0])

def add_transaction(user_id: int, kind: str, amount, category: str, description: str, occurred_on: str, recurring=False):
    if kind not in {"income", "expense"}: raise ValueError("Invalid transaction type.")
    date.fromisoformat(occurred_on)
    if not category.strip(): raise ValueError("Category is required.")
    with connect() as db:
        category_id=_category_id(db,user_id,category,kind)
        cur = db.execute("INSERT INTO transactions(user_id,kind,amount_minor,category,description,occurred_on,recurring,category_id) VALUES(?,?,?,?,?,?,?,?)",
                         (user_id, kind, to_minor(amount), category.strip(), description.strip(), occurred_on, int(recurring),category_id))
        return cur.lastrowid

def transactions(user_id: int, limit=500):
    with connect() as db:
        return db.execute("SELECT * FROM transactions WHERE user_id=? ORDER BY occurred_on DESC,id DESC LIMIT ?", (user_id, limit)).fetchall()

def delete_transaction(user_id: int, transaction_id: int) -> bool:
    with connect() as db:
        return db.execute("DELETE FROM transactions WHERE user_id=? AND id=?", (user_id, transaction_id)).rowcount == 1

def update_transaction(user_id: int, transaction_id: int, *, kind: str, amount, category: str, description: str, occurred_on: str, recurring=False) -> bool:
    if kind not in {"income", "expense"}: raise ValueError("Invalid transaction type.")
    date.fromisoformat(occurred_on)
    if not category.strip(): raise ValueError("Category is required.")
    with connect() as db:
        category_id=_category_id(db,user_id,category,kind)
        return db.execute("UPDATE transactions SET kind=?,amount_minor=?,category=?,description=?,occurred_on=?,recurring=?,category_id=? WHERE user_id=? AND id=?",
                          (kind,to_minor(amount),category.strip(),description.strip(),occurred_on,int(recurring),category_id,user_id,transaction_id)).rowcount == 1

def monthly_summary(user_id: int, month: str | None = None):
    month = month or date.today().strftime("%Y-%m")
    with connect() as db:
        rows = db.execute("SELECT kind,COALESCE(SUM(amount_minor),0) total FROM transactions WHERE user_id=? AND substr(occurred_on,1,7)=? GROUP BY kind",(user_id,month)).fetchall()
        result={r["kind"]:r["total"] for r in rows}
    return {"income":result.get("income",0),"expenses":result.get("expense",0),"surplus":result.get("income",0)-result.get("expense",0)}

def set_budget(user_id: int, category: str, amount):
    if not category.strip(): raise ValueError("Budget category is required.")
    with connect() as db:
        db.execute("INSERT INTO budgets(user_id,category,monthly_limit_minor) VALUES(?,?,?) ON CONFLICT(user_id,category) DO UPDATE SET monthly_limit_minor=excluded.monthly_limit_minor",(user_id,category.strip(),to_minor(amount)))

def budgets(user_id: int, month: str | None = None):
    month=month or date.today().strftime("%Y-%m")
    with connect() as db:
        return db.execute("SELECT b.category,b.monthly_limit_minor,COALESCE(SUM(t.amount_minor),0) spent_minor FROM budgets b LEFT JOIN transactions t ON t.user_id=b.user_id AND t.category=b.category AND t.kind='expense' AND substr(t.occurred_on,1,7)=? WHERE b.user_id=? GROUP BY b.id ORDER BY b.category",(month,user_id)).fetchall()

def add_goal(user_id: int, name: str, target, saved=0, due_on=None):
    saved_minor=int((Decimal(str(saved))*100).quantize(Decimal("1"),rounding=ROUND_HALF_UP))
    if not name.strip(): raise ValueError("Goal name is required.")
    if saved_minor < 0 or saved_minor > to_minor(target): raise ValueError("Saved amount must be between zero and the target.")
    if due_on: date.fromisoformat(due_on)
    with connect() as db:
        cur=db.execute("INSERT INTO goals(user_id,name,target_minor,saved_minor,due_on) VALUES(?,?,?,?,?)",(user_id,name.strip(),to_minor(target),saved_minor,due_on))
        return cur.lastrowid

def goals(user_id: int):
    with connect() as db: return db.execute("SELECT * FROM goals WHERE user_id=? ORDER BY id DESC",(user_id,)).fetchall()

def set_profile(user_id: int, income, emergency):
    income_minor=int((Decimal(str(income))*100).quantize(Decimal("1"),rounding=ROUND_HALF_UP))
    emergency_minor=int((Decimal(str(emergency))*100).quantize(Decimal("1"),rounding=ROUND_HALF_UP))
    if income_minor<0 or emergency_minor<0: raise ValueError("Profile amounts cannot be negative.")
    with connect() as db:
        db.execute("UPDATE profiles SET monthly_income_minor=?,emergency_fund_minor=? WHERE user_id=?",(income_minor,emergency_minor,user_id))

def profile(user_id: int):
    with connect() as db: return db.execute("SELECT * FROM profiles WHERE user_id=?",(user_id,)).fetchone()

def categories(user_id: int, kind: str | None = None):
    with connect() as db:
        if kind: return db.execute("SELECT * FROM categories WHERE user_id=? AND kind=? ORDER BY name",(user_id,kind)).fetchall()
        return db.execute("SELECT * FROM categories WHERE user_id=? ORDER BY kind,name",(user_id,)).fetchall()

def add_category(user_id: int, name: str, kind: str):
    if not name.strip() or kind not in {"income","expense"}: raise ValueError("Enter a category and choose income or expense.")
    with connect() as db: db.execute("INSERT OR IGNORE INTO categories(user_id,name,kind) VALUES(?,?,?)",(user_id,name.strip(),kind))

def add_recurring_obligation(user_id: int, name: str, category: str, amount, due_day: int):
    if not name.strip() or not category.strip() or not 1<=int(due_day)<=31: raise ValueError("Enter a name, category, and due day from 1 to 31.")
    with connect() as db:
        cur=db.execute("INSERT INTO recurring_obligations(user_id,name,category,amount_minor,due_day) VALUES(?,?,?,?,?)",(user_id,name.strip(),category.strip(),to_minor(amount),int(due_day)))
        return cur.lastrowid

def recurring_obligations(user_id: int, active_only=True):
    with connect() as db:
        query="SELECT * FROM recurring_obligations WHERE user_id=?"+(" AND active=1" if active_only else "")+" ORDER BY due_day,name"
        return db.execute(query,(user_id,)).fetchall()

def set_recurring_active(user_id: int, obligation_id: int, active: bool):
    with connect() as db: return db.execute("UPDATE recurring_obligations SET active=? WHERE user_id=? AND id=?",(int(active),user_id,obligation_id)).rowcount==1

def contribute_to_goal(user_id: int, goal_id: int, amount, contributed_on: str, note=""):
    date.fromisoformat(contributed_on); amount_minor=to_minor(amount)
    with connect() as db:
        goal=db.execute("SELECT * FROM goals WHERE id=? AND user_id=?",(goal_id,user_id)).fetchone()
        if not goal: raise ValueError("Savings goal was not found for this account.")
        current=db.execute("SELECT COALESCE(SUM(amount_minor),0) FROM goal_contributions WHERE goal_id=?",(goal_id,)).fetchone()[0]
        if goal["saved_minor"]+current+amount_minor>goal["target_minor"]: raise ValueError("Contribution would exceed the goal target.")
        db.execute("INSERT INTO goal_contributions(goal_id,amount_minor,contributed_on,note) VALUES(?,?,?,?)",(goal_id,amount_minor,contributed_on,note.strip()))

def goal_contributions(user_id: int, goal_id: int):
    with connect() as db:
        return db.execute("SELECT c.* FROM goal_contributions c JOIN goals g ON g.id=c.goal_id WHERE g.user_id=? AND g.id=? ORDER BY c.contributed_on DESC,c.id DESC",(user_id,goal_id)).fetchall()

def goals_with_progress(user_id: int):
    with connect() as db:
        return db.execute("SELECT g.*,g.saved_minor+COALESCE(SUM(c.amount_minor),0) AS total_saved_minor FROM goals g LEFT JOIN goal_contributions c ON c.goal_id=g.id WHERE g.user_id=? GROUP BY g.id ORDER BY g.id DESC",(user_id,)).fetchall()

def update_goal(user_id: int, goal_id: int, *, name: str, target, due_on=None):
    if due_on: date.fromisoformat(due_on)
    target_minor=to_minor(target)
    with connect() as db:
        goal=db.execute("SELECT saved_minor+COALESCE((SELECT SUM(amount_minor) FROM goal_contributions WHERE goal_id=goals.id),0) AS saved FROM goals WHERE id=? AND user_id=?",(goal_id,user_id)).fetchone()
        if not goal: return False
        if target_minor<goal["saved"]: raise ValueError("Goal target cannot be below the amount already saved.")
        return db.execute("UPDATE goals SET name=?,target_minor=?,due_on=? WHERE id=? AND user_id=?",(name.strip(),target_minor,due_on,goal_id,user_id)).rowcount==1

def delete_goal(user_id: int, goal_id: int):
    with connect() as db: return db.execute("DELETE FROM goals WHERE id=? AND user_id=?",(goal_id,user_id)).rowcount==1

def monthly_history(user_id: int, months=6):
    with connect() as db:
        rows=db.execute("SELECT substr(occurred_on,1,7) month,kind,SUM(amount_minor) total FROM transactions WHERE user_id=? GROUP BY month,kind ORDER BY month DESC LIMIT ?",(user_id,months*2)).fetchall()
    history={}
    for row in rows:
        history.setdefault(row["month"],{"income":0,"expenses":0,"surplus":0})
        key="income" if row["kind"]=="income" else "expenses"
        history[row["month"]][key]=row["total"]
    for value in history.values(): value["surplus"]=value["income"]-value["expenses"]
    return sorted(history.items())

def delete_budget(user_id: int, category: str):
    with connect() as db: return db.execute("DELETE FROM budgets WHERE user_id=? AND category=?",(user_id,category)).rowcount==1
