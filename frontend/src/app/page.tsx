import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import Link from "next/link"

export default function Home() {
  return (
    <div className="flex flex-col items-center min-h-[calc(100vh-4rem)]">
      <section className="flex max-w-[980px] flex-col items-center gap-8 py-24 px-4 text-center">
        <h1 className="text-3xl font-bold leading-tight tracking-tighter md:text-6xl lg:leading-[1.1] bg-clip-text text-transparent bg-gradient-to-b from-foreground to-foreground/70">
           Statistics & Fight Predictions
        </h1>
        <p className="max-w-[750px] text-lg text-muted-foreground sm:text-xl">
          Explore fighter statistics, rankings, and AI-powered fight predictions
        </p>
        <div className="flex gap-4">
          <Link href="/fighters">
            <Button size="lg" className="font-medium">View Fighters</Button>
          </Link>
          <Link href="/fight-predictions">
            <Button size="lg" variant="outline" className="font-medium">Fight Predictions</Button>
          </Link>
        </div>
      </section>

      <section className="w-full max-w-[1200px] px-4 pb-24">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          <Card className="bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Fighter Statistics</CardTitle>
              <CardDescription>
                Comprehensive stats for every UFC fighter
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">Access detailed fighter profiles including win/loss records, striking accuracy, grappling stats, and more.</p>
            </CardContent>
          </Card>

          <Card className="bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Fight Predictions</CardTitle>
              <CardDescription>
                AI-powered fight outcome predictions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">Get data-driven predictions for upcoming UFC fights based on historical performance and fighter matchups.</p>
            </CardContent>
          </Card>

          <Card className="bg-card/50 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Rankings</CardTitle>
              <CardDescription>
                Up-to-date UFC rankings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground">Stay informed with the latest UFC rankings across all weight divisions for both men and women.</p>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>
  )
}
