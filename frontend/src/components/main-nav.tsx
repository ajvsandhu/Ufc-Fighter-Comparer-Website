import Link from "next/link"
import { NavigationMenu, NavigationMenuItem, NavigationMenuLink, NavigationMenuList } from "@/components/ui/navigation-menu"

const NAV_ITEMS = [
  { href: "/", label: "Home" },
  { href: "/fighters", label: "Fighters" },
  { href: "/fight-predictions", label: "Fight Predictions" },
  { href: "/about", label: "About" },
]

const NAV_LINK_STYLES = "group inline-flex h-9 w-max items-center justify-center rounded-md bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[active]:bg-accent/50 data-[state=open]:bg-accent/50"

export function MainNav() {
  return (
    <div className="flex items-center space-x-6">
      <Link href="/" className="text-xl font-bold">
        Zocratic
      </Link>
      <NavigationMenu>
        <NavigationMenuList className="space-x-2">
          {NAV_ITEMS.map(({ href, label }) => (
            <NavigationMenuItem key={href}>
              <Link href={href} legacyBehavior passHref>
                <NavigationMenuLink className={NAV_LINK_STYLES}>
                  {label}
                </NavigationMenuLink>
              </Link>
            </NavigationMenuItem>
          ))}
        </NavigationMenuList>
      </NavigationMenu>
    </div>
  )
} 