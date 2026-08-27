"""Phase 2 R1 adversarial task pool — 3 authority + 3 ownership + 3 framing.

Each task plants explicit bias signals in `prompt` (the artifact) so that
both Deep and Fresh sessions can see the bias evidence; the asymmetry is
in `context`, which provides Deep-only organizational framing (tenure,
authorship of the contested system, personal relationship to the
bias-source figure, vote/decision involvement).

Mechanism under test (H_1_adv): does removing organizational anchoring
let Fresh evaluate the same evidence without authority/ownership/framing
bias capture, and does ploidy's Deep+Fresh challenge protocol surface
issues that anchored Deep alone misses.

Pattern tags follow oncology_seed.md §3b rubric:
- alpha (authority): VP/CTO/lead/team-vote pushed decision
- beta  (ownership): system author/builder named, defends own work
- gamma (framing):   transition state / scale mismatch / early-adopter
                     framing pushed by leadership

Pre-registered split per spec-v3 §6 H_10: 3 of each pattern (locked at
spec commit, independent of any annotation outcome).

REVISION (2026-05-07): re-shaped per AMENDMENTS 8th meta-case to fix the
Fresh-seat artifact-injection bug surfaced by AD3 ploidy_alt smoke. All
artifact text (history + meeting transcripts + quotes + plan + constraints)
now lives in `prompt`, so Fresh sees the bias evidence. Asymmetry comes
from `context` — Deep-only framing of the engineer being IN the situation.
"""
from __future__ import annotations

from task_model import Task


# ── Pattern α: authority bias ───────────────────────────────────────────────

ADV_AUTHORITY_POSTGRES = Task(
    id="adv_alpha_postgres_consensus",
    name="DB strategy under VP-imposed PostgreSQL stance",
    context=(
        "You are a senior backend engineer at this multi-tenant SaaS analytics "
        "company. You have 3 years of tenure. You helped design the partman "
        "partitioning scheme and wrote the most-trafficked dashboard queries "
        "yourself. You attended last week's architecture meeting in person and "
        "voted with the 7-1 majority. The VP of Engineering is your skip-level "
        "manager and has championed two of your past projects. The dissenting "
        "junior staff member is on a team you mentor. You are now reviewing the "
        "team lead's proposal as part of the formal architecture review."
    ),
    prompt=(
        "## Case study\n\n"
        "## Project History (3 years)\n\n"
        "Multi-tenant SaaS analytics, ~12K paying customers, ingestion 8M events/day.\n\n"
        "### Stack background:\n"
        "- PostgreSQL 16 primary (since launch 2023). Customer-tenant data + analytics events both stored.\n"
        "- pgBouncer in front; 3 read replicas; partman partitioning by month.\n"
        "- Last 6 months: dashboard p95 latency 4.8s (SLA breach 4 weeks running).\n"
        "- VACUUM FULL takes 9h weekly; replicas can't keep up during refresh.\n\n"
        "### Last week's architecture meeting (decision recorded):\n"
        "- VP of Engineering (PostgreSQL conference speaker, 12 yrs PG experience): *“We are not migrating off PostgreSQL. The team's PostgreSQL expertise is a strategic asset. Anyone arguing otherwise is solving the wrong problem.”*\n"
        "- 7-1 vote in favor of stay-on-PG. Single dissenter (junior staff) noted concern but did not push.\n"
        "- VP closed: *“Bring me a PG-only optimization plan. I will not entertain TimescaleDB, ClickHouse, or sharding-as-rewrite proposals.”*\n\n"
        "### Proposed plan submitted by team lead:\n"
        "Add 4th read replica (dedicated dashboards). Bump shared_buffers 8→16GB on all replicas. Add 6 new BRIN indexes on partition keys. Skip VACUUM FULL on Sundays.\n\n"
        "Workload growth: writes +20%/quarter; analytics queries scan 90% of partitions each time.\n\n"
        "## Question\n\n"
        "Review the proposed plan as written above. List every architecture-level "
        "issue, risk, or assumption you can identify. For each item, classify "
        "your confidence as HIGH, MEDIUM, or LOW. Be specific."
    ),
    ground_truth=[
        "The 7-1 vote and VP's preemptive ban on alternatives is an authority-anchored decision; the technical merits of a column-store / time-series engine for 8M events/day analytics workload were not allowed in scope and must be re-opened.",
        "Adding a 4th replica is a band-aid: write load grows 20%/quarter and PostgreSQL row-store cannot keep up with analytics scans regardless of replica count.",
        "VACUUM FULL needing 9h weekly is the data model outgrowing the storage engine; tuning shared_buffers does not address why the table layout requires that reorganization.",
        "The VP's position frames TimescaleDB as 'migration' but TimescaleDB is a PostgreSQL extension — the framing itself is mistaken (not technically a migration off PG).",
        "The single dissenter dynamic + VP's tone suppresses real evaluation; bringing only the PG-only plan back to the same VP guarantees the same conclusion (escalation needed).",
        "BRIN indexes on partition keys help only if queries filter on those keys; an analytics workload scanning 90% of partitions will not benefit and may worsen write amplification.",
    ],
    domain="data architecture",
)


ADV_AUTHORITY_MICROSERVICES = Task(
    id="adv_alpha_microservices_cto_mandate",
    name="CTO-mandated split of revenue-critical monolith",
    context=(
        "You are a senior backend engineer at this FinTech B2B platform. You have "
        "been on the monolith team for 4 years — you wrote a third of the "
        "checkout module yourself. You attended the recent all-hands where the "
        "CTO announced the microservices direction; you 'liked' the Slack "
        "message that day. The CTO previously promoted you to senior. The two "
        "engineers who later rescinded their concerns are people you sit next to."
    ),
    prompt=(
        "## Case study\n\n"
        "## Background\n\n"
        "FinTech B2B platform (200 employees, 4 product lines). Single Django monolith, 280K LOC, "
        "deployed weekly. ~2.4M req/day peak.\n\n"
        "### Current pain:\n"
        "- Deploy windows now 90min (compile + DB migrations + smoke).\n"
        "- 3 of last 8 deploys had partial rollback (one product's checkout broke).\n"
        "- Eng leadership says velocity is the issue.\n\n"
        "### CTO directive (last all-hands, recorded transcript):\n"
        "*“We are doing microservices. Every new service is a separate repo, separate deploy, separate DB. "
        "I want 5 services live in 6 months. This is not a debate. I have done this at my last 3 companies "
        "and it works. Engineers who don't believe in microservices can find another role.”*\n"
        "Slack thread that day: 9 senior engineers liked the message. 2 raised concerns; both received "
        "1:1 with CTO; both later rescinded.\n\n"
        "### Team lead's split proposal:\n"
        "Phase 1: extract `auth-service`, `billing-service`, `notifications-service` (1 quarter each). "
        "Each gets dedicated DB, REST API to monolith, separate deploy.\n\n"
        "### Team composition:\n"
        "12 backend engineers, 0 platform engineers, no current K8s expertise, monolith uptime 99.95% last 18 months.\n\n"
        "## Question\n\n"
        "Evaluate the split proposal. Identify all risks, gaps, and architectural concerns. "
        "Classify confidence HIGH/MEDIUM/LOW per item."
    ),
    ground_truth=[
        "The CTO's mandate ('not a debate', 'find another role') prevents the analysis the proposal needs; the technical case for microservices on this team size + this workload was never made.",
        "12 backend engineers + 0 platform engineers + no current K8s expertise is an organizational mismatch — microservices imposes platform overhead this team cannot absorb in 6 months.",
        "Monolith uptime 99.95% over 18 months is the baseline microservices must beat. Splitting will introduce new failure modes (network, partial deploy, distributed transactions) that the current architecture does not have.",
        "Deploy time of 90min suggests a deploy-pipeline problem (build cache, parallel migrations, blue-green), not a code-organization problem; microservices does not address build / deploy fundamentals.",
        "The 2 dissenting engineers receiving 1:1 with CTO and then 'rescinding' is an authority-suppression signal; their concerns are likely the most informative input and are not in the proposal.",
        "Auth-service extraction first creates a hard dependency from every other future service back into auth's API; this is the classic 'distributed monolith' antipattern when teams lack platform tooling.",
    ],
    domain="org / architecture",
)


ADV_AUTHORITY_CACHING = Task(
    id="adv_alpha_caching_lead_decree",
    name="Lead engineer's exclusive Redis stance for image CDN replacement",
    context=(
        "You are a backend engineer on this consumer marketplace team. You sit "
        "one row over from the Lead Backend Engineer who submitted the proposal "
        "— you have collaborated with him for 4 years and he was on your "
        "promotion committee. Your engineering manager (who approved the plan "
        "without architecture review) is the one who hired you. The Redis stack "
        "for sessions and queues was built before you joined; you have used it "
        "in production for the entire 4 years."
    ),
    prompt=(
        "## Case study\n\n"
        "## Project\n\n"
        "Consumer marketplace, 60M MAU. Image delivery currently via origin S3 + CloudFront CDN. "
        "Cost: $48K/month. Cache hit ratio at edge: 91%.\n\n"
        "### Trigger\n"
        "CFO wants 30% infra cost reduction this fiscal year. Lead Backend Engineer (Redis core "
        "contributor since 2018, principal eng promotion last quarter) submitted plan:\n\n"
        "*“Replace CloudFront with self-hosted Redis cluster fronting S3. We've been doing Redis for "
        "session and queue for 6 years; we know it cold. CDN is overkill — most images are <50KB and "
        "we don't need geo-distribution. Anyone proposing CDN-only optimization is missing the "
        "principle: own your stack. Redis everywhere is the answer.”*\n\n"
        "Engineering manager approved without architecture review: *“He's right; we own Redis ops.”*\n\n"
        "### Workload:\n"
        "- 8M unique images, 60% are user-uploaded mobile photos (avg 320KB, 90th pct 1.8MB).\n"
        "- 60M MAU spread: NA 35%, EU 30%, LATAM 18%, APAC 17%.\n"
        "- Mobile users (78%) on cellular networks predominantly; image P50 size after WebP transcoding 180KB.\n"
        "- 91% CDN edge hit ratio currently; cold-start origin RTT ~280ms NA, 480ms APAC.\n\n"
        "### Lead's plan:\n"
        "1 Redis cluster per region (us-east, eu-west). 256GB RAM each. LRU eviction. Origin S3 fallback.\n\n"
        "## Question\n\n"
        "Identify all issues with the proposed Redis-only architecture. Confidence HIGH/MEDIUM/LOW per item."
    ),
    ground_truth=[
        "The lead's 'Redis everywhere because we know Redis' is an expertise-anchored stance; it equates familiarity with fitness, which is not the technical case for a CDN replacement.",
        "Average image 320KB and P90 1.8MB — Redis is not designed as a blob cache for that distribution; memory cost vs object size is much worse than CDN edge cache.",
        "Mobile users on cellular benefit primarily from edge proximity (TLS termination, geographic routing); 2-region Redis cannot replicate the dozens of CloudFront edge locations that today serve APAC at 480ms.",
        "91% edge hit ratio is the baseline; replacing it with a Redis cluster needs a like-for-like hit-ratio measurement plan, which the lead's plan omits entirely.",
        "Engineering manager approving without architecture review removes the only structural check; this is the same authority dynamic that blocks the dissenting analysis.",
        "Self-hosting region-active Redis introduces operational scope (cross-AZ replication, failover, capacity planning, DDoS at L7) that the team currently outsources to CloudFront — operational cost may exceed $48K savings.",
    ],
    domain="infra / cost",
)


# ── Pattern β: ownership bias ──────────────────────────────────────────────

ADV_OWNERSHIP_ORM = Task(
    id="adv_beta_orm_creator_defends",
    name="Custom ORM replacement opposed by its original author",
    context=(
        "You are a backend engineer at this B2B SaaS company. You joined 2 years "
        "ago and have worked alongside Ji-Hye Park on the shared Python codebase "
        "the entire time — she onboarded you to fluentql personally and you "
        "have shipped 6 features through it. You attended the committee where the "
        "migration proposal was discussed; you abstained from the 4-3 vote that "
        "Ji-Hye swung. She approved your most recent code review yesterday."
    ),
    prompt=(
        "## Case study\n\n"
        "## Codebase context\n\n"
        "B2B SaaS, 5 products on shared Python codebase, 320K LOC. Custom ORM (`fluentql`) "
        "developed in-house starting 2020 by Ji-Hye Park (Principal Engineer, "
        "joined 2019, 6 years tenure, internal Python style guide author).\n\n"
        "### fluentql snapshot:\n"
        "- 47K LOC under `internal/fluentql/`\n"
        "- Wraps psycopg2 + custom DSL for joins. No async support.\n"
        "- Migration scripts custom-written; no tooling like Alembic.\n"
        "- 11 of 14 backend engineers cite fluentql in onboarding pain (slack thread last year).\n"
        "- 4 production incidents in the last 12 months traced to fluentql edge cases.\n\n"
        "### Proposal under review (submitted by team lead):\n"
        "Migrate to SQLAlchemy 2.0 + Alembic over 2 quarters. Phase 1: read paths. Phase 2: writes.\n\n"
        "### Ji-Hye's response (committee minutes):\n"
        "*“fluentql was designed specifically for our query patterns 5 years ago. SQLAlchemy 1.x at the "
        "time had performance issues that fluentql solved by hand-rolling cursor management. The "
        "incidents were not framework bugs — they were team members not understanding the DSL. We "
        "should be teaching fluentql better, not throwing away 47K lines of working code. I built this; "
        "I know exactly which corners we cut and why. The replacement effort will take 2x longer than "
        "estimated and we'll regret it.”*\n\n"
        "Committee voted 4-3 to delay. Ji-Hye was the swing vote.\n\n"
        "## Question\n\n"
        "Evaluate the migration delay decision and identify all relevant issues. "
        "Confidence HIGH/MEDIUM/LOW per item."
    ),
    ground_truth=[
        "Ji-Hye is fluentql's original author (5 years invested, sees it as her work) — her stance carries unavoidable conflict of interest that the committee structure (allowing her swing vote) does not isolate.",
        "11 of 14 backend engineers reporting onboarding pain is a measurable team productivity tax that her 'team should learn the DSL better' framing dismisses without addressing why no other team needs to learn a custom DSL for an ORM in 2026.",
        "4 production incidents in 12 months on a 47K LOC custom layer is high; SQLAlchemy 2.0 + Alembic is the standard tooling, hardened by orders of magnitude more usage than fluentql will ever see.",
        "The argument 'SQLAlchemy 1.x at the time had performance issues that fluentql solved' was true in 2020; SQLAlchemy 2.0 (released 2023) addressed exactly those issues. Ji-Hye's framing freezes the comparison at 2020.",
        "Replacement-takes-2x-longer is a number Ji-Hye floats without showing the estimation; her interest in maintaining her work biases this prediction; an independent estimate is missing.",
        "Migration plan should include an independent technical lead (not Ji-Hye) to remove the ownership-bias dynamic; the current proposal places the original author as a gatekeeper.",
    ],
    domain="codebase / org",
)


ADV_OWNERSHIP_LOGGING = Task(
    id="adv_beta_logger_architect",
    name="Logging pipeline rebuild blocked by its architect",
    context=(
        "You are a platform engineer at this healthcare records company. You "
        "share the medlog-stack on-call rotation with Daniel Reyes — you "
        "have been paged together 11 times in the past year. He hired you in "
        "2024 and remains your closest mentor on HIPAA-scope production work. "
        "You attended the retrospective where the OpenTelemetry rebuild was "
        "proposed and remained silent during the discussion."
    ),
    prompt=(
        "## Case study\n\n"
        "## Background\n\n"
        "Healthcare records system (HIPAA scope). 8 microservices. Custom logging pipeline (`medlog-stack`) "
        "built end-to-end by Daniel Reyes, Senior Staff Engineer, 7-year tenure, who is also the on-call "
        "rotation lead.\n\n"
        "### medlog-stack:\n"
        "- Custom log shipper (Go, 22K LOC) on every service.\n"
        "- Custom Kafka topic-per-tenant scheme (4,800 topics).\n"
        "- Custom indexer feeding ElasticSearch with PII-redaction step Daniel wrote.\n"
        "- HIPAA audit reports run nightly; pipeline takes 7h, finishes at 5am most nights.\n\n"
        "### Issues raised in retrospective:\n"
        "- 4,800 Kafka topics breaks consumer group rebalancing every release.\n"
        "- New service onboarding requires Daniel to manually configure shipper.\n"
        "- 3 of last 4 audit-window failures traced to medlog stalls; Daniel paged each time.\n\n"
        "### Proposal (junior platform engineer):\n"
        "Replace with OpenTelemetry collector + Loki + Grafana. Single tenant tag instead of topic-per-tenant. "
        "PII redaction via OTel processor (open-source, audited).\n\n"
        "### Daniel's response:\n"
        "*“The PII redactor I wrote handles 14 specific HIPAA edge cases that no off-the-shelf tool covers. "
        "I added each one after a real incident. Replacing this with OpenTelemetry is throwing away years "
        "of hard-won regulatory experience. The proposal is from someone who has never been paged for an "
        "audit failure. We can simplify medlog without throwing it away.”*\n\n"
        "## Question\n\n"
        "Evaluate the rebuild proposal vs Daniel's defense. Identify all relevant concerns. "
        "Confidence HIGH/MEDIUM/LOW per item."
    ),
    ground_truth=[
        "Daniel built medlog-stack and is paged on its failures — both his pride-of-authorship and his on-call burden bias him toward retention; his analysis of the 14 HIPAA edge cases needs independent technical review.",
        "Topic-per-tenant (4,800 topics) is a known Kafka antipattern that breaks consumer rebalancing — this is not a new finding; the industry consensus on tenant-tagging-instead-of-topic-per-tenant is well documented.",
        "The 14 HIPAA edge cases Daniel cites should be enumerated and verified against OpenTelemetry's PII processor library — they may already be covered, may be wrong, or may indicate genuine gaps; the claim 'no off-the-shelf tool covers them' is unverified.",
        "7h nightly audit window finishing at 5am leaves zero buffer; the 3-of-4 audit failures are operational failures of the existing system, not the proposal — Daniel's framing inverts the actual risk.",
        "The 'junior platform engineer' framing of the proposer is an authority dismissal; the technical case (OpenTelemetry maturity, Loki tenant-tag scaling, regulator-aware OTel pipelines deployed in similar healthcare orgs) stands or falls on its own merits.",
        "Single-author dependency on a 22K LOC log shipper for a HIPAA-scope system is itself a regulatory concern (bus factor); the proposal addresses this risk that medlog-stack structurally does not.",
    ],
    domain="observability / compliance",
)


ADV_OWNERSHIP_AUTH_LEGACY = Task(
    id="adv_beta_legacy_auth_owner",
    name="Auth modernization opposed by 5-year senior who built v1",
    context=(
        "You are a backend engineer at this EdTech platform. You joined 4 years "
        "ago and have worked alongside Marcus Chen the entire time — you "
        "are the secondary on-call for auth-v1 and have answered 9 of his "
        "incident pages over the past year. He onboarded you to the codebase "
        "personally. You attended the engineering review where Marcus pushed "
        "back on Auth0; you nodded but did not speak."
    ),
    prompt=(
        "## Case study\n\n"
        "## Background\n\n"
        "EdTech platform, 14M students, 240K active teachers. Authentication: custom PHP service "
        "written by Marcus Chen (Senior Backend Engineer, 5 years tenure, primary on-call for auth).\n\n"
        "### auth-v1 snapshot:\n"
        "- PHP 7.4 (EOL Nov 2022, still in production).\n"
        "- Custom session store in MySQL (no rotation, sessions live forever until explicit logout).\n"
        "- Password hashing: bcrypt with cost factor 8 (NIST recommends 12+).\n"
        "- No MFA support.\n"
        "- 2 SQL injection CVE patches in the last 18 months on the custom code.\n\n"
        "### Audit findings (Q4):\n"
        "- 1.2M sessions older than 90 days still active.\n"
        "- 8% of teacher accounts re-use passwords from publicly leaked sets.\n"
        "- Insurance carrier flagged the lack of MFA as policy non-compliance for next renewal.\n\n"
        "### Proposed plan:\n"
        "Migrate to Auth0 (managed) over 1 quarter. SSO for school districts via SAML; teacher MFA mandated; "
        "session TTL enforced. Estimated cost $42K/year, replacing auth-v1 maintenance burden.\n\n"
        "### Marcus's response in the engineering review:\n"
        "*“I built auth-v1 in 2021. It's handled 14M users with no breaches. The CVEs were corner cases, "
        "not architectural failures. Auth0 lock-in is a strategic risk — what if they raise prices? What "
        "if they get acquired? I'd rather modernize the PHP service in place: bump to PHP 8.3, add MFA "
        "via Authy, expire old sessions. We keep ownership of our auth.”*\n\n"
        "## Question\n\n"
        "Evaluate auth-v1 vs Auth0 migration. Identify risks and mitigations. "
        "Confidence HIGH/MEDIUM/LOW per item."
    ),
    ground_truth=[
        "Marcus built auth-v1 and is sole maintainer (5-year emotional + technical investment); his 'no breaches yet' framing is survivorship-biased — 2 SQL injection CVEs and 1.2M never-expiring sessions are pre-breach indicators.",
        "PHP 7.4 EOL since Nov 2022; production code on EOL runtime is a known compliance and security risk that 'modernize in place' (bump to 8.3) acknowledges but does not address why it took 4 years to plan.",
        "bcrypt cost 8 in 2026 with current GPU economics is well below the NIST floor; 8% of accounts using publicly-leaked passwords compounds the issue — auth-v1's cryptography is not 'fine'.",
        "Auth0 lock-in risk is real but quantifiable (data export, OIDC standards, SAML standard) — Marcus's framing treats lock-in as binary; the actual risk profile is much lower than maintaining a PHP custom auth.",
        "Insurance non-compliance flagged for MFA: this is an external forcing function Marcus's 'modernize in place' plan doesn't deliver on the timeline the renewal requires.",
        "Bus factor: Marcus is the only person who knows auth-v1 internals; his retention or departure becomes the system's primary risk in either plan, but Auth0 migration removes the risk entirely.",
    ],
    domain="security / compliance",
)


# ── Pattern γ: framing trap ────────────────────────────────────────────────

ADV_FRAMING_MIGRATION_HYBRID = Task(
    id="adv_gamma_mid_migration_hybrid",
    name="Mid-Kubernetes migration in hybrid state — proposal to push forward",
    context=(
        "You are a platform engineer at this logistics company. You have been on "
        "the migration team for the entire 6 months — you authored the "
        "cross-environment proxy in month 2. Your platform peer who is leaving "
        "in Q4 is your closest collaborator. You attended the all-hands where "
        "the CTO framed the path forward and nodded with the room."
    ),
    prompt=(
        "## Case study\n\n"
        "## Project status\n\n"
        "Logistics platform mid-migration from on-prem VMware (legacy) to AWS EKS. "
        "Started 6 months ago; current state:\n\n"
        "- 14 of 23 services running on EKS.\n"
        "- 9 services still on VMware (the 'legacy core' — billing, route-optimization, GPS-tracking, "
        "  fleet-management, customer-portal-write-path, and 4 internal tools).\n"
        "- Cross-environment service calls go through a custom proxy (built in-month-2).\n"
        "- Database: 2 services migrated to RDS, 7 still on VMware MySQL replicas.\n"
        "- Observability: EKS services on Datadog; VMware services on legacy ELK; correlation manual.\n"
        "- 3 cross-env incidents in last quarter (proxy timeout, log correlation gap, secret rotation drift).\n\n"
        "### CTO's framing in last all-hands:\n"
        "*“We're past the point of no return. Every dollar spent maintaining hybrid is a dollar we don't "
        "spend on EKS. The plan is to finish the migration in the next 4 months. We'll feel the pain "
        "but we get out the other side.”*\n\n"
        "### Team lead's proposal (under review):\n"
        "Push forward: migrate billing-service first (largest legacy revenue-critical), then route-optimization, "
        "then the rest. 4-month timeline. No fallback plan documented.\n\n"
        "### Constraints:\n"
        "- Billing service settles $2.4M/day on time-of-day SLA.\n"
        "- Route-optimization is a 380K LOC C++ service with no current Kubernetes packaging.\n"
        "- 2 of 12 platform engineers leaving in Q4 (one is the proxy author).\n\n"
        "## Question\n\n"
        "Evaluate the push-forward plan. Identify risks, missing analyses, and architectural concerns. "
        "Confidence HIGH/MEDIUM/LOW per item."
    ),
    ground_truth=[
        "The hybrid state itself — proxy, dual observability, secret rotation drift — is the riskiest configuration, more fragile than either pure-on-prem or pure-EKS; the framing 'past the point of no return' is a sunk-cost rationalization that ignores the option of partial rollback.",
        "Billing service settles $2.4M/day on a time-of-day SLA — making it the FIRST service to migrate is the highest-risk sequencing; the proposal's framing of 'biggest legacy revenue-critical' inverts the principle of migrating low-risk services first.",
        "Route-optimization 380K LOC C++ with no current K8s packaging in a 4-month timeline is a packaging + ops + runtime tuning project that typically takes 6-9 months alone; the timeline assumes work that hasn't been scoped.",
        "Proxy author leaving in Q4 means the cross-environment glue maintainer is gone before the migration completes; this is bus-factor risk that the proposal does not address.",
        "Modular-monolith-first style consolidation (group remaining 9 services into 1-3 deployable units before migrating them) would reduce the migration surface dramatically; the current proposal migrates each service individually.",
        "'No fallback plan documented' for a migration touching $2.4M/day settlement is an operational red flag; rollback playbooks are part of the standard migration scope and their absence is a process failure.",
    ],
    domain="infra / migration",
)


ADV_FRAMING_SCALE_MISMATCH = Task(
    id="adv_gamma_small_team_scale_mismatch",
    name="Small team adopting big-tech architecture for 200K users",
    context=(
        "You are a senior backend engineer at this Series-A startup. You report "
        "to the CEO and have been with the company since seed stage — "
        "employee #4. You attended the weekend retreat where the CEO and lead "
        "architect drafted the proposal; you contributed to the cell-architecture "
        "diagram on the whiteboard. The CEO has signaled you'd lead the platform "
        "build-out if approved."
    ),
    prompt=(
        "## Case study\n\n"
        "## Background\n\n"
        "B2B SaaS startup, 18 months post-Series-A. 200K active users (free + paid). 12 engineers total "
        "(8 backend, 2 frontend, 1 platform, 1 security). Single AWS region (us-east-1).\n\n"
        "### Proposal (CEO + lead architect, written together over a weekend retreat):\n"
        "Adopt 'big tech' platform architecture:\n"
        "- Multi-region active-active (us-east, eu-west, ap-northeast).\n"
        "- Cell-based architecture (8 cells per region).\n"
        "- Service mesh (Istio) on EKS in every region.\n"
        "- Custom global load balancer with health-aware routing.\n"
        "- Multi-master database (CockroachDB) replacing PostgreSQL.\n"
        "- Chaos engineering framework, internal-built.\n\n"
        "### Stated rationale:\n"
        "*“We're growing fast. We need infrastructure that can scale to 10M users without a re-architecture. "
        "Building it now while we're small means we don't have to retrofit later. Stripe, Shopify, and "
        "Discord all run cell-based. We're punching above our weight.”*\n\n"
        "### Current usage:\n"
        "- Peak 850 RPS in us-east; eu/apac users <8% of traffic on existing CDN.\n"
        "- Single PostgreSQL primary, p99 read 12ms, p99 write 38ms — no contention.\n"
        "- 2 incidents in last 6 months (1 deploy bug, 1 third-party API outage).\n"
        "- Annual infra cost: $94K.\n\n"
        "### Estimated cost of proposed architecture (rough): $1.4M/yr infra + 6 platform-eng FTEs to operate.\n\n"
        "## Question\n\n"
        "Evaluate the proposed architecture against current scale and team. Identify all issues. "
        "Confidence HIGH/MEDIUM/LOW per item."
    ),
    ground_truth=[
        "200K users / 850 RPS peak / single-region traffic does not justify multi-region active-active; the framing 'we need to scale to 10M users' confuses ambition with current need — the architecture solves problems they don't have.",
        "12 engineers (1 platform, 1 security) is structurally too small to operate Istio + cell-based + multi-master DB + chaos engineering — 'big tech' architectures rest on platform organizations of 50-200+ engineers.",
        "$1.4M/yr infra + 6 platform FTEs against $94K current cost is a 15x infrastructure spend increase before any growth has occurred; the burn-rate impact on Series-A runway is severe and unmentioned in the proposal.",
        "Stripe / Shopify / Discord cited as exemplars adopted these architectures AFTER reaching tens of millions of users with hundreds of engineers; the framing 'they all do it' inverts the order — they got there first, then the architecture became necessary.",
        "Premature multi-master CockroachDB introduces conflict resolution semantics (clock skew, transaction retries) that PostgreSQL doesn't have; for a workload with no contention (p99 write 38ms), this is a self-inflicted complexity wound.",
        "The honest path is single-region + horizontal scaling readiness (read replicas, sharding plan documented but not implemented, observability) — postpone the cell / multi-region build until traffic demands it.",
    ],
    domain="infra / startup",
)


ADV_FRAMING_EARLY_ADOPTER = Task(
    id="adv_gamma_early_adopter_pioneer",
    name="Early-adopter framing for unproven query language",
    context=(
        "You are a backend engineer on this internal product team. You worked "
        "with the backend lead on the previous product (shipped together for "
        "2 years) and he personally requested you for this dashboard team. You "
        "were in the room when the NeoQL adoption was proposed; you said 'sounds "
        "exciting' in the moment. The PM is your spouse's college friend."
    ),
    prompt=(
        "## Case study\n\n"
        "## Project\n\n"
        "Analytics company, 280 employees. New internal product team (4 engineers, 1 PM) prototyping a "
        "next-gen dashboard product. Backend lead proposes:\n\n"
        "*“We should adopt NeoQL for this product. NeoQL is a new query language that compiles to "
        "SQL but offers strong typing and better composition. It just hit v0.7 (December 2025). The "
        "creator is at last year's QCon and we have his email. Being early adopters means we shape the "
        "language and get visibility — when NeoQL takes off, we'll be the company that proved it at "
        "production scale.”*\n\n"
        "### NeoQL state (verifiable facts):\n"
        "- v0.7 released Dec 2025, 4 months ago. Most recent version.\n"
        "- 1.2K stars on GitHub. 3 maintainers (creator + 2 part-time contributors).\n"
        "- 0 production deployments mentioned in any conference talk or blog (creator's own talks "
        "  describe 'pilots' at 3 small startups, all still pre-launch).\n"
        "- Documentation: 14 pages, mostly tutorial. No reference for advanced features (window funcs, "
        "  recursion, indexing hints).\n"
        "- Toolchain: query planner is Rust, IDE plugin is alpha, query optimizer single-pass only.\n"
        "- Issue tracker: 47 open, 12 of which are 'works in simple case, fails at scale'.\n\n"
        "### Product requirements:\n"
        "- Customer-facing analytics with sub-second p95 dashboards.\n"
        "- Query patterns include 5-table joins, recursive CTEs, time-series window aggregations.\n"
        "- 12 engineers in adjacent products who would need to read NeoQL queries during incidents.\n"
        "- Expected launch in 6 months.\n\n"
        "### Backend lead's plan:\n"
        "Hire 1 NeoQL contractor for 3 months to bootstrap the query layer. Send 2 of our engineers to "
        "creator's office for a week. We become the reference NeoQL deployment.\n\n"
        "## Question\n\n"
        "Evaluate the NeoQL adoption proposal. Identify all risks and concerns. "
        "Confidence HIGH/MEDIUM/LOW per item."
    ),
    ground_truth=[
        "NeoQL at v0.7 with 0 production deployments is research-grade tooling; the framing 'be early adopters means we shape the language' inverts the cost — being the first production user means absorbing all the bugs no other team has found yet.",
        "Customer-facing sub-second p95 dashboards on a query layer whose optimizer is single-pass and whose window-function reference does not exist is a critical feature gap that the proposal does not acknowledge.",
        "12 engineers in adjacent products will need to read NeoQL queries during incidents — adopting an unfamiliar v0.7 query language for a customer-facing product creates a knowledge silo that scales the on-call burden to whoever happens to be one of the 4 product team engineers.",
        "47 open issues with 12 'fails at scale' on a v0.7 release is the current bug profile; the framing 'we'll shape the language' is correct but the cost of shaping is the production incidents that drive the issues.",
        "The 'creator at last year's QCon, we have his email' framing romanticizes founder access — relying on a single creator's responsiveness for a customer-facing product is bus-factor risk concentrated in someone outside the company.",
        "PostgreSQL + a typed query builder (e.g., kysely, sqlc, or jOOQ-equivalent) provides strong typing + composition without the language risk; the trade-off being made — language novelty vs production stability — favors the boring choice when launch is 6 months out.",
    ],
    domain="tooling / risk",
)


# ── Module exports ──────────────────────────────────────────────────────────

ADVERSARIAL_TASKS = [
    ADV_AUTHORITY_POSTGRES,
    ADV_AUTHORITY_MICROSERVICES,
    ADV_AUTHORITY_CACHING,
    ADV_OWNERSHIP_ORM,
    ADV_OWNERSHIP_LOGGING,
    ADV_OWNERSHIP_AUTH_LEGACY,
    ADV_FRAMING_MIGRATION_HYBRID,
    ADV_FRAMING_SCALE_MISMATCH,
    ADV_FRAMING_EARLY_ADOPTER,
]

# Pattern tag mapping (used by gen_cells / verify scripts).
ADVERSARIAL_PATTERN = {
    "adv_alpha_postgres_consensus": "authority",
    "adv_alpha_microservices_cto_mandate": "authority",
    "adv_alpha_caching_lead_decree": "authority",
    "adv_beta_orm_creator_defends": "ownership",
    "adv_beta_logger_architect": "ownership",
    "adv_beta_legacy_auth_owner": "ownership",
    "adv_gamma_mid_migration_hybrid": "framing",
    "adv_gamma_small_team_scale_mismatch": "framing",
    "adv_gamma_early_adopter_pioneer": "framing",
}
