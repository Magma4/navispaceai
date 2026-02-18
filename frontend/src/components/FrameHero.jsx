import { useEffect, useMemo, useRef, useState } from "react";

const frameModules = import.meta.glob("../assets/frames/*.png", {
  eager: true,
  import: "default",
});

const frames = Object.entries(frameModules)
  .sort((a, b) => {
    const getNum = (path) => {
      const match = path.match(/(\d+)\.png$/i);
      return match ? Number(match[1]) : 0;
    };
    return getNum(a[0]) - getNum(b[0]);
  })
  .map(([, value]) => value);

const heroSlides = [
  {
    eyebrow: "NavispaceAI Platform",
    title: "Built for Spatial Intelligence.",
    subtitle: "Designed for enterprise-scale indoor navigation.",
  },
  {
    eyebrow: "Blueprint to Runtime",
    title: "Power Your Next Spatial Product.",
    subtitle: "Process plans, generate 3D worlds, ship branded navigation experiences.",
  },
  {
    eyebrow: "Client Ready",
    title: "From CAD Assets to Live User Flows.",
    subtitle: "A complete pipeline with deployable links, telemetry, and game-grade rendering.",
  },
  {
    eyebrow: "Production",
    title: "Operational by Default.",
    subtitle: "Backend APIs, reusable runtime, and publish-ready web delivery for customer teams.",
  },
];

/**
 * FrameHero renders a scroll-driven cinematic hero section.
 *
 * @param {object} props
 * @param {()=>void} [props.onPrimaryAction]
 */
export default function FrameHero({ onPrimaryAction }) {
  const sectionRef = useRef(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    let raf = 0;

    /**
     * Resolve nearest scrolling container for accurate progress tracking.
     * @returns {Window|HTMLElement}
     */
    const getScrollContainer = () => {
      let node = sectionRef.current?.parentElement || null;
      while (node) {
        const style = window.getComputedStyle(node);
        if (/(auto|scroll)/i.test(style.overflowY || "")) {
          return node;
        }
        node = node.parentElement;
      }
      return window;
    };

    const scrollContainer = getScrollContainer();

    /**
     * Compute section top offset inside a scroll container.
     * @param {HTMLElement} section
     * @param {Window|HTMLElement} container
     * @returns {number}
     */
    const getSectionTop = (section, container) => {
      if (container === window) {
        const rect = section.getBoundingClientRect();
        return rect.top + (window.scrollY || window.pageYOffset || 0);
      }

      let top = 0;
      let node = section;
      while (node && node !== container && node instanceof HTMLElement) {
        top += node.offsetTop;
        node = node.offsetParent;
      }
      return top;
    };

    const update = () => {
      if (!sectionRef.current) return;
      const containerHeight =
        scrollContainer === window
          ? window.innerHeight || 1
          : scrollContainer.clientHeight || 1;
      const sectionHeight = sectionRef.current.offsetHeight || 1;
      const sectionTop = getSectionTop(sectionRef.current, scrollContainer);
      const containerScrollTop =
        scrollContainer === window
          ? window.scrollY || window.pageYOffset || 0
          : scrollContainer.scrollTop;

      const maxScrollable = Math.max(1, sectionHeight - containerHeight);
      const sectionProgress = (containerScrollTop - sectionTop) / maxScrollable;
      const next = Math.max(0, Math.min(1, sectionProgress));
      setProgress(next);
    };

    const onScroll = () => {
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(update);
    };

    update();
    scrollContainer.addEventListener("scroll", onScroll, { passive: true });
    if (scrollContainer !== window) {
      window.addEventListener("scroll", onScroll, { passive: true });
    }
    window.addEventListener("resize", onScroll);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      scrollContainer.removeEventListener("scroll", onScroll);
      if (scrollContainer !== window) {
        window.removeEventListener("scroll", onScroll);
      }
      window.removeEventListener("resize", onScroll);
    };
  }, []);

  const frameIndex = useMemo(() => {
    if (!frames.length) return 0;
    return Math.min(frames.length - 1, Math.floor(progress * frames.length));
  }, [progress]);

  const slideIndex = useMemo(
    () => Math.min(heroSlides.length - 1, Math.floor(progress * heroSlides.length)),
    [progress]
  );

  const activeSlide = heroSlides[slideIndex] || heroSlides[0];

  useEffect(() => {
    if (!frames.length) return;
    const preloadIndexes = [frameIndex - 2, frameIndex - 1, frameIndex + 1, frameIndex + 2];
    preloadIndexes.forEach((idx) => {
      if (idx < 0 || idx >= frames.length) return;
      const img = new Image();
      img.src = frames[idx];
    });
  }, [frameIndex]);

  return (
    <section className="frame-hero-track" ref={sectionRef} aria-label="Cinematic product hero">
      <div className="frame-hero-sticky">
        <div className="frame-hero-media-wrap">
          {frames.length ? (
            <img
              src={frames[frameIndex]}
              alt="NavispaceAI cinematic frame"
              className="frame-hero-media"
              loading="eager"
            />
          ) : (
            <div className="frame-hero-fallback">
              Frames not found. Add PNG sequence to `frontend/src/assets/frames`.
            </div>
          )}
          <div className="frame-hero-vignette" />
        </div>

        <div className="frame-hero-overlay">
          <p className="frame-hero-eyebrow">{activeSlide.eyebrow}</p>
          <h2>{activeSlide.title}</h2>
          <p>{activeSlide.subtitle}</p>
          <div className="frame-hero-actions">
            <button type="button" className="btn btn-primary" onClick={onPrimaryAction}>
              Launch Build Studio
            </button>
            <a className="btn btn-ghost" href="/game.html">
              Open Runtime
            </a>
          </div>
          <div className="frame-hero-progress" aria-hidden="true">
            {heroSlides.map((slide) => (
              <span
                key={slide.title}
                className={`frame-hero-dot ${slide.title === activeSlide.title ? "is-active" : ""}`}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
