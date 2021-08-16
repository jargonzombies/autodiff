#pragma once

#include <autodiff/reverse/var/var.hpp>

#include <vector>
#include <unordered_map>
#include <string>
#include <sstream>

namespace autodiff {
namespace detail {

template<typename T>
struct GraphvizVisitor {
  // The pointer to the expression is a unique vertex descriptor
  using Pointer = Expr<T>*;

  bool register_node(Pointer t, const std::string& name) {
    if(vertices.count(t) == 0) {
      vertices.emplace(t, name);
      return true;
    }

    return false;
  }

  template<typename U>
  void descend(Pointer from, U* to, const char* name) {
    visit(to);
    edges.emplace_back(from, to, name);
  }

  void visit(Pointer t) {
    if(auto independent = dynamic_cast<IndependentVariableExpr<T>*>(t)) {
      if(register_node(t, "Independent")) {
        descend(t, independent->gradx.get(), "gradx");
      }
      return;
    }

    if(auto dependent = dynamic_cast<DependentVariableExpr<T>*>(t)) {
      if(register_node(t, "Dependent")) {
        descend(t, dependent->expr.get(), "expr");
        descend(t, dependent->gradx.get(), "gradx");
      }
      return;
    }

    if(auto constant = dynamic_cast<ConstantExpr<T>*>(t)) {
      register_node(t, std::to_string(constant->val));
      return;
    }

    if(auto unary = dynamic_cast<NegativeExpr<T>*>(t)) {
      if(register_node(t, "Negate")) {
        descend(t, unary->x.get(), "x");
      }
      return;
    }

    if(auto add = dynamic_cast<AddExpr<T>*>(t)) {
      if(register_node(t, "Add")) {
        descend(t, add->l.get(), "x");
        descend(t, add->r.get(), "y");
      }
      return;
    }

    if(auto sub = dynamic_cast<SubExpr<T>*>(t)) {
      if(register_node(t, "Sub")) {
        descend(t, sub->l.get(), "x");
        descend(t, sub->r.get(), "y");
      }
      return;
    }

    if(auto mul = dynamic_cast<MulExpr<T>*>(t)) {
      if(register_node(t, "Mul")) {
        descend(t, mul->l.get(), "x");
        descend(t, mul->r.get(), "y");
      }
      return;
    }

    if(auto div = dynamic_cast<DivExpr<T>*>(t)) {
      if(register_node(t, "Div")) {
        descend(t, div->l.get(), "x");
        descend(t, div->r.get(), "y");
      }
      return;
    }

    if(auto sin = dynamic_cast<SinExpr<T>*>(t)) {
      if(register_node(t, "Sin")) {
        descend(t, sin->x.get(), "x");
      }
      return;
    }

    if(auto cos = dynamic_cast<CosExpr<T>*>(t)) {
      if(register_node(t, "Cos")) {
        descend(t, cos->x.get(), "x");
      }
      return;
    }

    if(auto tan = dynamic_cast<TanExpr<T>*>(t)) {
      if(register_node(t, "Tan")) {
        descend(t, tan->x.get(), "x");
      }
      return;
    }

    if(auto sinh = dynamic_cast<SinhExpr<T>*>(t)) {
      if(register_node(t, "Sinh")) {
        descend(t, sinh->x.get(), "x");
      }
      return;
    }

    if(auto cosh = dynamic_cast<CoshExpr<T>*>(t)) {
      if(register_node(t, "Cosh")) {
        descend(t, cosh->x.get(), "x");
      }
      return;
    }

    if(auto tanh = dynamic_cast<TanhExpr<T>*>(t)) {
      if(register_node(t, "Tanh")) {
        descend(t, tanh->x.get(), "x");
      }
      return;
    }

    if(auto arcsin = dynamic_cast<ArcSinExpr<T>*>(t)) {
      if(register_node(t, "Arcsin")) {
        descend(t, arcsin->x.get(), "x");
      }
      return;
    }

    if(auto arccos = dynamic_cast<ArcCosExpr<T>*>(t)) {
      if(register_node(t, "ArcCos")) {
        descend(t, arccos->x.get(), "x");
      }
      return;
    }

    if(auto arctan = dynamic_cast<ArcTanExpr<T>*>(t)) {
      if(register_node(t, "ArcTan")) {
        descend(t, arctan->x.get(), "x");
      }
      return;
    }

    if(auto arctan2 = dynamic_cast<ArcTan2Expr<T>*>(t)) {
      if(register_node(t, "ArcTan2")) {
        descend(t, arctan2->l.get(), "x");
        descend(t, arctan2->r.get(), "y");
      }
      return;
    }

    if(auto exp = dynamic_cast<ExpExpr<T>*>(t)) {
      if(register_node(t, "Exp")) {
        descend(t, exp->x.get(), "x");
      }
      return;
    }

    if(auto log = dynamic_cast<LogExpr<T>*>(t)) {
      if(register_node(t, "Log")) {
        descend(t, log->x.get(), "x");
      }
      return;
    }

    if(auto log10 = dynamic_cast<Log10Expr<T>*>(t)) {
      if(register_node(t, "Log10")) {
        descend(t, log10->x.get(), "x");
      }
      return;
    }

    if(auto pow = dynamic_cast<PowExpr<T>*>(t)) {
      if(register_node(t, "Pow")) {
        descend(t, pow->l.get(), "x");
        descend(t, pow->r.get(), "y");
      }
      return;
    }

    if(auto pow = dynamic_cast<PowConstantLeftExpr<T>*>(t)) {
      if(register_node(t, "PowConstX")) {
        descend(t, pow->l.get(), "x");
        descend(t, pow->r.get(), "y");
      }
      return;
    }

    if(auto pow = dynamic_cast<PowConstantRightExpr<T>*>(t)) {
      if(register_node(t, "PowConstY")) {
        descend(t, pow->l.get(), "x");
        descend(t, pow->r.get(), "y");
      }
      return;
    }

    if(auto sqrt = dynamic_cast<SqrtExpr<T>*>(t)) {
      if(register_node(t, "Sqrt")) {
        descend(t, sqrt->x.get(), "x");
      }
      return;
    }

    if(auto abs = dynamic_cast<AbsExpr<T>*>(t)) {
      if(register_node(t, "Abs")) {
        descend(t, abs->x.get(), "x");
      }
      return;
    }

    if(auto hypot2 = dynamic_cast<Hypot2Expr<T>*>(t)) {
      if(register_node(t, "Hypot2")) {
        descend(t, hypot2->l.get(), "x");
        descend(t, hypot2->r.get(), "y");
      }
      return;
    }

    if(auto hypot3 = dynamic_cast<Hypot3Expr<T>*>(t)) {
      if(register_node(t, "Hypot3")) {
        descend(t, hypot3->l.get(), "x");
        descend(t, hypot3->c.get(), "y");
        descend(t, hypot3->r.get(), "z");
      }
      return;
    }

    register_node(t, "Unknown");
  }

  void dump(std::ostream& viz) const {
    viz << "digraph G {\n";
    for(const auto& [ptr, name] : vertices) {
      viz << "  E" << ptr << " [label=\"" << name << "\"]\n";
    };
    for(const auto& [from, to, label] : edges) {
      viz << "  E" << from << " -> E" << to << " [label=\"" << label << "\"]\n";
    }
    viz << "}\n";
  }

  std::unordered_map<Pointer, std::string> vertices;
  std::vector<std::tuple<Pointer, Pointer, std::string>> edges;
};

} // namespace detail

template<typename T>
std::ostream& graphviz(std::ostream& os, detail::ExprPtr<T> expr) {
  detail::GraphvizVisitor<T> visitor;
  visitor.visit(expr.get());
  visitor.dump(os);
  return os;
}

template<typename T>
std::string graphviz(detail::ExprPtr<T> expr) {
  detail::GraphvizVisitor<T> visitor;
  visitor.visit(expr.get());
  std::stringstream viz;
  visitor.dump(viz);
  return viz.str();
}

} // namespace autodiff
