import {
  Activity,
  CircleHelp,
  Compass,
  Home,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

const navigation = [
  { label: "Home", href: "#home", icon: Home },
  { label: "About", href: "#about", icon: Compass },
  { label: "Features", href: "#features", icon: Sparkles },
  { label: "Analytics", href: "#analytics", icon: Activity },
  { label: "Help", href: "#help", icon: CircleHelp },
];

interface SidebarProps {
  activeSection: string;
  onNavigate: (section: string) => void;
}

export function Sidebar({ activeSection, onNavigate }: SidebarProps) {
  return (
    <aside className="sidebar">
      <a className="sidebar-brand" href="#home" onClick={() => onNavigate("home")}>
        <span className="brand-mark" aria-hidden="true">
          <ShieldCheck size={23} strokeWidth={2.2} />
        </span>
        <span className="brand-name">
          VERIFAI
          <small>IMAGE INTELLIGENCE</small>
        </span>
      </a>

      <p className="sidebar-label">WORKSPACE</p>
      <nav className="side-nav" aria-label="Main navigation">
        {navigation.map(({ label, href, icon: Icon }) => {
          const section = href.slice(1);
          return (
            <a
              key={section}
              href={href}
              className={`nav-link${activeSection === section ? " is-active" : ""}`}
              aria-current={activeSection === section ? "page" : undefined}
              onClick={() => onNavigate(section)}
            >
              <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
              <span>{label}</span>
            </a>
          );
        })}
      </nav>

      <div className="sidebar-note">
        <div className="sidebar-note-icon">
          <Sparkles size={16} aria-hidden="true" />
        </div>
        <strong>Look beyond the surface</strong>
        <p>AI-assisted screening, with the evidence kept in view.</p>
      </div>

      <div className="sidebar-footer">
        <span className="sidebar-footer-dot" />
        Research preview
      </div>
    </aside>
  );
}
