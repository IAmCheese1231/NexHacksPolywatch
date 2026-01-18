"use client";
import React, { useMemo, useState } from "react";
import {
  Share2,
  Wallet,
  Settings,
  GitBranch,
  PieChart,
  LineChart,
  BarChart3,
  Search,
  Plus,
  Trash2,
  Play,
  Check,
  X,
} from "lucide-react";

/**
 * Scenario Analyzer (frontend mock)
 * - React + Tailwind
 * - Uses lucide-react icons
 * - Single-file component
 *
 * Fix included: children are OPTIONAL for Card/Pill (prevents TS2741).
 */

type Market = {
  id: string;
  name: string;
  prob: number; // 0..1
  category: string;
  tag: string;
};

type NodeSim = {
  id: string;
  name: string;
  baseFrom: number; // 0..1
  baseTo: number; // 0..1
};

const fmtPct = (p: number) => `${Math.round(p * 100)}%`;

function Pill({
  children,
  tone = "neutral",
}: {
  children?: React.ReactNode; // ✅ optional
  tone?: "neutral" | "green" | "red" | "blue";
}) {
  if (!children) return null; // ✅ don't render empty pill
  const cls =
    tone === "green"
      ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/30"
      : tone === "red"
      ? "bg-rose-500/15 text-rose-300 border-rose-500/30"
      : tone === "blue"
      ? "bg-sky-500/15 text-sky-200 border-sky-500/30"
      : "bg-white/5 text-slate-200 border-white/10";

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium ${cls}`}
    >
      {children}
    </span>
  );
}

function Card({
  children,
  className = "",
}: {
  children?: React.ReactNode; // ✅ optional
  className?: string;
}) {
  return (
    <div
      className={`rounded-2xl border border-white/10 bg-white/5 shadow-[0_10px_40px_-20px_rgba(0,0,0,0.6)] backdrop-blur ${className}`}
    >
      {children}
    </div>
  );
}

function CardHeader({
  title,
  right,
  icon,
}: {
  title: string;
  right?: React.ReactNode;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 border-b border-white/10 px-5 py-4">
      <div className="flex items-center gap-3">
        {icon ? (
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-white/5 text-sky-200">
            {icon}
          </div>
        ) : null}
        <div className="text-lg font-semibold tracking-tight text-slate-100">
          {title}
        </div>
      </div>
      {right ? <div className="flex items-center gap-2">{right}</div> : null}
    </div>
  );
}

function Stat({
  label,
  value,
  accent = "neutral",
}: {
  label: string;
  value: string;
  accent?: "neutral" | "green" | "blue";
}) {
  const valueCls =
    accent === "green"
      ? "text-emerald-300"
      : accent === "blue"
      ? "text-sky-200"
      : "text-slate-100";
  return (
    <Card className="px-5 py-4">
      <div className="text-xs font-medium text-slate-300/80">{label}</div>
      <div className={`mt-2 text-2xl font-semibold ${valueCls}`}>{value}</div>
    </Card>
  );
}

function Progress({
  label,
  valueLabel,
  value,
  tone = "blue",
}: {
  label: string;
  valueLabel: string;
  value: number; // 0..1
  tone?: "blue" | "green" | "neutral";
}) {
  const bar =
    tone === "green"
      ? "bg-emerald-400/60"
      : tone === "blue"
      ? "bg-sky-400/60"
      : "bg-white/30";
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold text-slate-100">{label}</div>
        <div className="text-sm font-semibold text-slate-200">{valueLabel}</div>
      </div>
      <div className="mt-3 h-2 w-full rounded-full bg-white/10">
        <div
          className={`h-2 rounded-full ${bar}`}
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
      </div>
    </div>
  );
}

export default function ScenarioAnalyzerUI() {
  const [activeTab, setActiveTab] = useState<
    "Scenario Builder" | "Portfolio" | "Market Explorer" | "Analytics"
  >("Scenario Builder");

  const [search, setSearch] = useState("");
  const [connected, setConnected] = useState(false);

  const markets: Market[] = [
    {
      id: "m1",
      name: "Trump Wins 2024",
      prob: 0.48,
      category: "US Election",
      tag: "Politics",
    },
    {
      id: "m2",
      name: "Fed Rate Cut Q1",
      prob: 0.62,
      category: "Macro",
      tag: "Macro",
    },
    {
      id: "m3",
      name: "Bitcoin > $100k",
      prob: 0.35,
      category: "Crypto",
      tag: "Crypto",
    },
    {
      id: "m4",
      name: "AI Regulation Passed",
      prob: 0.41,
      category: "AI/Tech",
      tag: "Tech",
    },
  ];

  const filtered = useMemo(() => {
    const s = search.trim().toLowerCase();
    if (!s) return markets;
    return markets.filter(
      (m) =>
        m.name.toLowerCase().includes(s) ||
        m.category.toLowerCase().includes(s) ||
        m.tag.toLowerCase().includes(s)
    );
  }, [search]);

  const [tree, setTree] = useState<NodeSim[]>([
    { id: "root", name: "Trump Wins 2024", baseFrom: 0.48, baseTo: 0.55 },
  ]);

  const root = tree[0];

  const correlated = [
    { id: "c1", name: "Fed Rate Cut", from: 0.62, to: 0.71, delta: +0.09 },
    { id: "c2", name: "Bitcoin > $100k", from: 0.35, to: 0.28, delta: -0.07 },
  ];

  const ev = 847;
  const winProb = 0.673;
  const sharpe = 1.82;

  return (
    <div className="min-h-screen bg-[#070b17] text-slate-100">
      {/* soft background glow */}
      <div className="pointer-events-none fixed inset-0">
        <div className="absolute left-1/3 top-[-10%] h-[520px] w-[520px] rounded-full bg-sky-500/10 blur-[80px]" />
        <div className="absolute right-[-10%] top-1/3 h-[520px] w-[520px] rounded-full bg-violet-500/10 blur-[90px]" />
      </div>

      <div className="relative mx-auto max-w-6xl px-5 pb-10 pt-7">
        {/* Top bar */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-11 w-11 place-items-center rounded-2xl bg-sky-500/15 text-sky-200 ring-1 ring-sky-500/20">
              <Share2 className="h-6 w-6" />
            </div>
            <div>
              <div className="text-2xl font-semibold tracking-tight">
                Scenario Analyzer
              </div>
              <div className="text-sm text-slate-300/80">
                TradFi-style scenario PnL for Polymarket portfolios
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setConnected((v) => !v)}
              className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-semibold text-slate-100 shadow-sm hover:bg-white/10"
            >
              <Wallet className="h-4 w-4 text-slate-200" />
              {connected ? "Wallet Connected" : "Connect Wallet"}
            </button>
            <button className="grid h-11 w-11 place-items-center rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10">
              <Settings className="h-5 w-5 text-slate-200" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="mt-6 flex items-center gap-2 border-b border-white/10">
          {[
            { key: "Scenario Builder", icon: <GitBranch className="h-4 w-4" /> },
            { key: "Portfolio", icon: <PieChart className="h-4 w-4" /> },
            { key: "Market Explorer", icon: <LineChart className="h-4 w-4" /> },
            { key: "Analytics", icon: <BarChart3 className="h-4 w-4" /> },
          ].map((t) => {
            const on = activeTab === (t.key as any);
            return (
              <button
                key={t.key}
                onClick={() => setActiveTab(t.key as any)}
                className={`relative -mb-px inline-flex items-center gap-2 px-4 py-3 text-sm font-semibold ${
                  on
                    ? "text-sky-200"
                    : "text-slate-300/80 hover:text-slate-200"
                }`}
              >
                <span
                  className={`${on ? "text-sky-200" : "text-slate-300/80"}`}
                >
                  {t.icon}
                </span>
                {t.key}
                {on ? (
                  <span className="absolute inset-x-3 bottom-0 h-[2px] rounded-full bg-sky-400/70" />
                ) : null}
              </button>
            );
          })}
        </div>

        {/* Main grid */}
        <div className="mt-6 grid grid-cols-12 gap-5">
          {/* Left: Event library */}
          <Card className="col-span-12 lg:col-span-3">
            <CardHeader
              title="Event Library"
              right={
                <button className="grid h-9 w-9 place-items-center rounded-xl border border-white/10 bg-white/5 hover:bg-white/10">
                  <Plus className="h-4 w-4 text-slate-200" />
                </button>
              }
            />
            <div className="px-5 py-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-300/70" />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search market"
                  className="w-full rounded-2xl border border-white/10 bg-white/5 py-2.5 pl-10 pr-3 text-sm text-slate-100 placeholder:text-slate-300/60 outline-none focus:border-sky-400/40"
                />
              </div>

              <div className="mt-4 space-y-3">
                {filtered.map((m) => (
                  <button
                    key={m.id}
                    className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-left hover:bg-white/10"
                    onClick={() => {
                      // quick add: creates a node in the tree (demo)
                      setTree((prev) => {
                        if (prev.some((n) => n.name === m.name)) return prev;
                        return [
                          ...prev,
                          {
                            id: m.id,
                            name: m.name,
                            baseFrom: m.prob,
                            baseTo: m.prob,
                          },
                        ];
                      });
                    }}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="text-sm font-semibold text-slate-100">
                        {m.name}
                      </div>
                      <div className="text-sm font-semibold text-slate-200">
                        {fmtPct(m.prob)}
                      </div>
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <Pill tone="blue">{m.category}</Pill>
                      <Pill>{m.tag}</Pill>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </Card>

          {/* Center: Scenario tree */}
          <Card className="col-span-12 lg:col-span-6">
            <CardHeader
              title="Scenario Tree"
              right={
                <>
                  <button
                    className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm font-semibold text-slate-200 hover:bg-white/10"
                    onClick={() =>
                      setTree([
                        {
                          id: "root",
                          name: "Trump Wins 2024",
                          baseFrom: 0.48,
                          baseTo: 0.55,
                        },
                      ])
                    }
                  >
                    <Trash2 className="h-4 w-4" />
                    Clear
                  </button>
                  <button
                    className="inline-flex items-center gap-2 rounded-xl border border-sky-400/30 bg-sky-500/20 px-3 py-2 text-sm font-semibold text-sky-100 hover:bg-sky-500/25"
                    onClick={() => {
                      // hook up your real simulate action here
                    }}
                  >
                    <Play className="h-4 w-4" />
                    Simulate
                  </button>
                </>
              }
            />
            <div className="px-5 py-5">
              <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-[#060a14] p-6">
                {/* Root node */}
                <div className="mx-auto w-full max-w-md">
                  <div className="rounded-2xl border border-white/10 bg-gradient-to-r from-sky-500/40 via-violet-500/35 to-fuchsia-500/35 px-6 py-6 text-center shadow-[0_20px_60px_-30px_rgba(0,0,0,0.9)]">
                    <div className="text-xl font-semibold">{root.name}</div>
                    <div className="mt-2 text-sm text-slate-100/80">
                      Base: {fmtPct(root.baseFrom)} →{" "}
                      <span className="font-semibold text-slate-100">
                        {fmtPct(root.baseTo)}
                      </span>
                    </div>

                    <div className="mt-4 flex items-center justify-center gap-4">
                      <Pill tone="green">
                        <Check className="h-3.5 w-3.5" /> YES
                      </Pill>
                      <Pill tone="red">
                        <X className="h-3.5 w-3.5" /> NO
                      </Pill>
                    </div>
                  </div>
                </div>

                {/* Correlated nodes */}
                <div className="mt-7 grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div className="rounded-2xl border border-emerald-500/25 bg-emerald-500/5 p-4">
                    <div className="text-sm font-semibold text-slate-100">
                      {correlated[0].name}
                    </div>
                    <div className="mt-1 text-sm text-slate-200/80">
                      {fmtPct(correlated[0].from)} →{" "}
                      <span className="font-semibold text-slate-100">
                        {fmtPct(correlated[0].to)}
                      </span>
                    </div>
                    <div className="mt-2 text-sm font-semibold text-emerald-300">
                      +{Math.round(correlated[0].delta * 100)}% (correlation)
                    </div>
                  </div>

                  <div className="rounded-2xl border border-rose-500/25 bg-rose-500/5 p-4">
                    <div className="text-sm font-semibold text-slate-100">
                      {correlated[1].name}
                    </div>
                    <div className="mt-1 text-sm text-slate-200/80">
                      {fmtPct(correlated[1].from)} →{" "}
                      <span className="font-semibold text-slate-100">
                        {fmtPct(correlated[1].to)}
                      </span>
                    </div>
                    <div className="mt-2 text-sm font-semibold text-rose-300">
                      {Math.round(correlated[1].delta * 100)}% (correlation)
                    </div>
                  </div>
                </div>

                <div className="mt-6 flex items-center justify-center">
                  <button className="inline-flex items-center gap-2 rounded-2xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-semibold text-slate-200 hover:bg-white/10">
                    <Plus className="h-4 w-4" />
                    Add Event Node
                  </button>
                </div>

                <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-slate-300/80">
                  <span className="mr-2">💡</span>
                  Tip: Drag events from the library to build your scenario tree.
                  Adjust probabilities with sliders to see correlated market
                  movements.
                </div>
              </div>
            </div>
          </Card>

          {/* Right: Scenario results */}
          <div className="col-span-12 lg:col-span-3 space-y-5">
            <Card>
              <CardHeader
                title="Scenario Results"
                icon={<BarChart3 className="h-5 w-5" />}
              />
              <div className="space-y-3 px-5 py-4">
                <Stat label="Expected Value" value={`+$${ev}`} accent="green" />
                <Stat
                  label="Win Probability"
                  value={`${(winProb * 100).toFixed(1)}%`}
                  accent="blue"
                />
                <Stat label="Sharpe Ratio" value={sharpe.toFixed(2)} />
              </div>
            </Card>

            <Card>
              <CardHeader
                title="Payoff Distribution"
                icon={<LineChart className="h-5 w-5" />}
              />
              <div className="space-y-3 px-5 py-4">
                <Progress
                  label="Best Case"
                  valueLabel="+$2,400"
                  value={0.12}
                  tone="green"
                />
                <Progress
                  label="Base Case"
                  valueLabel="+$850"
                  value={0.58}
                  tone="blue"
                />
                <Progress
                  label="Worst Case"
                  valueLabel="-$900"
                  value={0.3}
                  tone="neutral"
                />
              </div>
            </Card>
          </div>
        </div>

        <div className="mt-8 text-xs text-slate-300/60">
          UI mock only — wire up Polymarket data + scenario engine behind the
          “Simulate” button.
        </div>
      </div>
    </div>
  );
}
